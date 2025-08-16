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
    # 在训练两个阶段时，BN 层都置为 eval
    if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
        m.eval()

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
        D_hidden = 512
        feat_dim = self.base_network.output_num
        self.domain_discriminator = nn.Sequential(
            nn.Linear(feat_dim, D_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(D_hidden, 1),
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
    def forward_stage2(self, mid, target, mid_label, mid_weight=None):
        """
        阶段二：中间域→目标域对抗对齐
        返回: (clf_loss, transfer_loss)
        """
        # 冻结 BN
        self.base_network.apply(fix_bn)

        # 1) mid 有监督分类损失
        feat_m = self.base_network.forward_features(mid)
        ce_none = nn.CrossEntropyLoss(reduction="none", label_smoothing=self.args.label_smoothing)
        per = ce_none(self.classifier_layer(feat_m), mid_label)  # [B]
        if mid_weight is not None:
            # 可选样本权重
            if not torch.is_tensor(mid_weight):
                mid_weight = torch.tensor(mid_weight, dtype=per.dtype, device=per.device)
            else:
                mid_weight = mid_weight.to(per.device, dtype=per.dtype)
            per = per * mid_weight
        loss_m = per.mean()  # -> 作为 clf_loss 返回

        # 2) 域对抗对齐损失
        # 1.特征提取
        feat_t = self.base_network.forward_features(target)
        # 2.GRL处理
        grl_m = grad_reverse(feat_m, self.args.lambda2)
        grl_t = grad_reverse(feat_t, self.args.lambda2)
        # 3.送入域判别器
        Dm = self.domain_discriminator(grl_m).view(-1)
        Dt = self.domain_discriminator(grl_t).view(-1)

        # 4. 计算域对抗损失
        bce = nn.BCEWithLogitsLoss() # 二元交叉熵 (BCE)损失
        loss_adv = 0.5 * (bce(Dm, torch.zeros_like(Dm)) + bce(Dt, torch.ones_like(Dt)))

        transfer_loss = self.args.lambda3 * loss_adv
        # rain() 里按 clf_loss + transfer_loss 相加
        return loss_m, transfer_loss

    def get_parameters(self, initial_lr=1.0):
        return [
            {"params": self.base_network.model.visual.parameters(),
             "lr": initial_lr},
            {"params": self.classifier_layer.parameters(),
             "lr": self.args.multiple_lr_classifier * initial_lr},
            {"params": self.domain_discriminator.parameters(),
             "lr": initial_lr * 0.1},
        ]

    def predict(self, x):
        feat = self.base_network.forward_features(x)
        return self.classifier_layer(feat)

    def clip_predict(self, x):
        return self.base_network(x)
