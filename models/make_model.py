# make_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from models.backbone import get_backbone
from models import cmkd
from utils.grad_reverse import grad_reverse

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        nn.init.normal_(m.weight, std=0.001)
    elif classname.find("BatchNorm") != -1:
        m.bias.requires_grad_(False)
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def fix_bn(m):
    if isinstance(m, nn.BatchNorm2d):
        m.eval()  # 固定 running mean/var
        m.requires_grad_(True)  # 保持可学习的 gamma/beta

class TransferNet(nn.Module):
    def __init__(self, args, train=True):
        super(TransferNet, self).__init__()
        self.args = args
        # backbone + classifier
        self.base_network = get_backbone(args).cuda()

        self.classifier_layer = nn.Sequential(
            nn.BatchNorm1d(self.base_network.output_num),
            nn.LayerNorm(self.base_network.output_num, eps=1e-6),
            nn.Linear(self.base_network.output_num, args.num_class, bias=False),
        )
        self.classifier_layer.apply(weights_init_classifier)
        # 对抗判别器
        # D_hidden = 512
        feat_dim = self.base_network.output_num
        # self.domain_discriminator = nn.Sequential(
        #     nn.Linear(feat_dim, D_hidden),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(D_hidden, 1),
        # )
        # self.domain_discriminator = nn.Sequential(
        #     nn.Linear(feat_dim, 512),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(0.5),
        #     nn.Linear(512, 256),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(0.5),
        #     nn.Linear(256, 1),
        # )
        # self.domain_discriminator = nn.Sequential(
        #     nn.Linear(feat_dim, 512),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(512, 1),
        # )
        self.domain_discriminator = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),  # 使用LeakyReLU而不是ReLU
            nn.Linear(256, 1),
        )

        # 损失
        if train:
            self.clf_loss = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
            self.cmkd     = cmkd.CMKD(args)

    def forward_stage1(self, source, mid, source_label, mid_label):
        """
        阶段一：有监督分类（source → mid）
        loss =  L_ce(source) + λ1 * L_ce(mid)
        """
        # 冻结 BN
        self.base_network.apply(fix_bn)

        # source 分类
        feat_s = self.base_network.forward_features(source)
        loss_s = self.clf_loss(self.classifier_layer(feat_s), source_label)

        # mid 分类
        feat_m = self.base_network.forward_features(mid)
        ce_none = nn.CrossEntropyLoss(reduction="none", label_smoothing=self.args.label_smoothing)
        per = ce_none(self.classifier_layer(feat_m), mid_label)
        loss_m = per.mean()

        total = loss_s + self.args.lambda1 * loss_m
        return total, loss_s.item(), loss_m.item()

    # make_model.py 中
    def forward_stage2(self, mid, target, mid_label, mid_weight=None, lambda2=None):
        """
        阶段二：中间域 → 目标域对抗对齐
        Args:
            mid: 中间域图像
            target: 目标域图像
            mid_label: 中间域标签
            mid_weight: 样本权重 (可选, Tensor 或 list)
            lambda2: 当前动态 λ2 (可选, float)，用于 warm-up

        Returns:
            clf_loss: 分类损失 (mid)
            transfer_loss: 域对齐损失
            Dm, Dt: 判别器输出 (便于调试打印)
        """
        # ====== 固定 BN ======
        self.base_network.apply(fix_bn)

        # ====== 1) mid 分类损失 ======
        feat_m = self.base_network.forward_features(mid)
        ce_none = nn.CrossEntropyLoss(reduction="none", label_smoothing=self.args.label_smoothing)
        per = ce_none(self.classifier_layer(feat_m), mid_label)

        if mid_weight is not None:
            # 确保 mid_weight 与 per dtype/device 一致
            if not torch.is_tensor(mid_weight):
                mid_weight = torch.tensor(mid_weight, dtype=per.dtype, device=per.device)
            else:
                mid_weight = mid_weight.to(per.device, dtype=per.dtype)
            # 保证 shape 对齐
            if mid_weight.shape[0] == per.shape[0]:
                per = per * mid_weight

        clf_loss = per.mean()

        # ====== 2) 域对抗对齐 ======
        feat_t = self.base_network.forward_features(target)

        # λ2 动态选择
        lambda2_now = lambda2 if lambda2 is not None else self.args.lambda2

        # GRL
        grl_m = grad_reverse(feat_m, lambda2_now)
        grl_t = grad_reverse(feat_t, lambda2_now)

        # 判别器输出
        Dm = self.domain_discriminator(grl_m).view(-1)
        Dt = self.domain_discriminator(grl_t).view(-1)

        # BCE loss
        bce = nn.BCEWithLogitsLoss()
        loss_adv = 0.5 * (
                bce(Dm, torch.zeros_like(Dm)) +  # mid → 0
                bce(Dt, torch.ones_like(Dt))  # target → 1
        )

        # λ3 缩放的 transfer_loss
        transfer_loss = self.args.lambda3 * loss_adv

        return clf_loss, transfer_loss, Dm, Dt

    def get_parameters(self, initial_lr=1.0):
        return [
            {"params": self.base_network.model.visual.parameters(),"lr": initial_lr},
            {"params": self.classifier_layer.parameters(), "lr": self.args.multiple_lr_classifier * initial_lr},
            {"params": self.domain_discriminator.parameters(),"lr": initial_lr * 100.0},
        ]

    def predict(self, x):
        feat = self.base_network.forward_features(x)
        return self.classifier_layer(feat)

    def clip_predict(self, x):
        return self.base_network(x)
