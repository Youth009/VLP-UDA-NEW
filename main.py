import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import torch.multiprocessing
import torch.nn.functional as F  # 在文件顶部添加
torch.multiprocessing.set_sharing_strategy('file_system')
import  copy
import itertools
import time
import torch
import ssl
import random
from sklearn import metrics
import numpy as np
from tqdm import tqdm
import configargparse
from torch.utils.data import DataLoader, WeightedRandomSampler
from typing import Optional
from utils import data_loader
from utils.build_dataloaders import build_dataloaders
from utils.tools import str2bool, AverageMeter, save_model
from models.make_model import TransferNet, fix_bn
import torchvision
from torchvision import datasets, transforms
import os
from models import rst
import logging
# from utils.mid_dataset import MidDatasetWithWeight
import json
from torch.cuda.amp import GradScaler, autocast
ssl._create_default_https_context = ssl._create_unverified_context

scaler = GradScaler()
def get_parser():
    """Get default arguments."""
    parser = configargparse.ArgumentParser(
        description="Transfer learning config parser",
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        formatter_class=configargparse.ArgumentDefaultsHelpFormatter,
    )
    # general configuration
    parser.add("--config", is_config_file=True, help="config file path")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--log_dir', type=str, default='log')
    parser.add_argument('--datasets', type=str, default='office_home',choices=["office_home","phyto_plankton","office31","visda",
                                                                               "domain_net","digits","image_clef"])
    parser.add_argument('--use_amp', type=str2bool, default=True)
    parser.add_argument('--d_steps', type=int, default=1, help='Number of discriminator steps per generator step')

    # network related
    parser.add_argument('--model_name', type=str, default='RN50',choices=["RN50", "VIT-B", "RN101"])

    # data loading related
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--src_domain', type=str, required=True)
    parser.add_argument('--mid_domain', type=str, required=True)
    parser.add_argument('--tgt_domain', type=str, required=True)

    # training related
    parser.add_argument('--l_batch_size', type=int, default=32)
    parser.add_argument('--u_batch_size', type=int, default=32)
    parser.add_argument('--n_epoch', type=int, default=20)
    parser.add_argument('--label_smoothing', type=float, default=0.0)
    parser.add_argument("--n_iter_per_epoch", type=int, default=500, help="Used in Iteration-based training")
    parser.add_argument('--rst_threshold', type=float, default=1e-5)
    parser.add_argument('--baseline', default=False, action='store_true')
    parser.add_argument('--pda', default=False, action='store_true')
    parser.add_argument('--rst', default=False, action='store_true')
    parser.add_argument('--clip', default=False, action='store_true')

    # FixMatch
    parser.add_argument('--fixmatch', default=False, action='store_true')
    parser.add_argument('--fixmatch_threshold', type=float, default=0.95)
    parser.add_argument('--fixmatch_factor', type=float, default=0.5)

    # optimizer related
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--lr_d', type=float, default=None, help='Discriminator learning rate')
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--multiple_lr_classifier', type=float, default=10)

    # loss related
    parser.add_argument('--lambda1', type=float, default=0.25)
    parser.add_argument('--lambda2', type=float, default=0.1)
    parser.add_argument('--lambda3', type=float, default=0.025)
    parser.add_argument('--clf_loss', type=str, default="cross_entropy")
    ### MODIFY ### 新增阶段划分参数
    parser.add_argument('--stage1_epochs', type=int, default=1,
                        help='前多少个 epoch 用"阶段一"训练')
    # learning rate scheduler related
    parser.add_argument('--scheduler', type=str2bool, default=True)
    parser.add_argument('--warmup_ratio', type=float, default=0.3,help='lambda2 的 warm-up 比例(0~1)')

    # linear scheduler
    parser.add_argument('--lr_gamma', type=float, default=0.0003)
    parser.add_argument('--lr_decay', type=float, default=0.75)

    return parser

def set_random_seed(seed=0):
    # seed setting
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_data(args):
    '''
    src_domain, tgt_domain data to load
    '''
    # Use FixMatch
    use_fixmatch = args.fixmatch
    folder_src = os.path.join(args.data_dir, args.src_domain)
    folder_mid = os.path.join(args.data_dir, args.mid_domain)
    folder_tgt = os.path.join(args.data_dir, args.tgt_domain)
    source_loader, n_class = data_loader.load_data(
        args, folder_src, args.l_batch_size, infinite_data_loader=True, train=True, num_workers=args.num_workers)
    mid_loader, _ = data_loader.load_data(
        args, folder_mid, args.l_batch_size, infinite_data_loader=True, train=True, num_workers=args.num_workers)
    # with open("mid_weight.json") as f:
    #          weight_dict = json.load(f)
    # weight_dict = {
    #     os.path.basename(path): 1.0 for path, _ in mid_loader.dataset.samples}
    # 用带权重的 Dataset 包装它
    # mid_ds = MidDatasetWithWeight(mid_loader.dataset, weight_dict)
    from torch.utils.data import DataLoader
    # mid_loader = DataLoader(
    #     mid_ds,
    #     batch_size=args.l_batch_size,
    #     shuffle=True,
    #     num_workers=args.num_workers,
    #     drop_last=True
    # )
    target_train_loader, _ = data_loader.load_data(
        args, folder_tgt, args.u_batch_size, infinite_data_loader=True, train=True, use_fixmatch=use_fixmatch, num_workers=args.num_workers, partial=args.pda)
    target_test_loader, _ = data_loader.load_data(
        args, folder_tgt, args.u_batch_size, infinite_data_loader=False, train=False, num_workers=args.num_workers, partial=args.pda)
    return source_loader, mid_loader, target_train_loader, target_test_loader, n_class

def get_model(args):
    model = TransferNet(args).to(args.device)
    print(f"模型所在设备: {next(model.parameters()).device}")
    return model

# def get_optimizer(model, args):
#     initial_lr = args.lr if not args.scheduler else 1.0
#     params = model.get_parameters(initial_lr=initial_lr)
#     optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
#     return optimizer

def get_optimizers(model, args):
    lr_g = args.lr_g if getattr(args, "lr_g", None) is not None else args.lr
    lr_d = args.lr_d if getattr(args, "lr_d", None) is not None else args.lr * 0.1

    g_params = [
        # backbone（CLIP visual）
        {"params": model.base_network.model.visual.parameters(), "lr": lr_g},
        # 分类头（一定要加进来！）
        {"params": model.classifier_layer.parameters(),
         "lr": lr_g * args.multiple_lr_classifier},
    ]
    optimizer_g = torch.optim.SGD(
        g_params, lr=lr_g, momentum=args.momentum,
        weight_decay=args.weight_decay, nesterov=True
    )

    optimizer_d = torch.optim.Adam(
        model.domain_discriminator.parameters(),
        lr=lr_d, betas=(0.9, 0.999), weight_decay=1e-4
    )
    return optimizer_g, optimizer_d

# def get_lr_scheduler(optimizer, args):
#     scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda x:  (args.lr * (1. + args.lr_gamma * float(x)) ** (-args.lr_decay)))
#     return scheduler

def get_lr_scheduler(optimizer_g, args):
    # 常见写法：随着 step 增加，lr 下降
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer_g,
        lr_lambda=lambda step: (1. + args.lr_gamma * float(step)) ** (-args.lr_decay)
    )

def test(model, target_test_loader, args):
    model.eval()
    test_loss = AverageMeter()
    criterion = torch.nn.CrossEntropyLoss()
    first_test = True
    desc = "Clip Testing..." if args.clip else "Testing..."
    with torch.no_grad():
        for data, target in tqdm(iterable=target_test_loader,desc=desc):
            data, target = data.to(args.device), target.to(args.device)
            if args.clip:
                s_output = model.clip_predict(data)
            else:
                s_output = model.predict(data)
            loss = criterion(s_output, target)
            test_loss.update(loss.item())
            pred = torch.max(s_output, 1)[1]
            if first_test:
                all_pred = pred
                all_label = target
                first_test = False
            else:
                all_pred = torch.cat((all_pred, pred), 0)
                all_label = torch.cat((all_label, target), 0)

    if args.datasets == "visda":
        acc = metrics.balanced_accuracy_score(all_label.cpu().numpy(),
                                                          torch.squeeze(all_pred).float().cpu().numpy()) *100
        cm = metrics.confusion_matrix(all_label.cpu().numpy(),
                                              torch.squeeze(all_pred).float().cpu().numpy())
        per_classes_acc = list(((cm.diagonal() / cm.sum(1))*100).round(4))
        per_classes_acc = list(map(str, per_classes_acc))
        per_classes_acc = ', '.join(per_classes_acc)
        if args.clip:
            print('CLIP: test_loss {:4f}, test_acc: {:.4f} \nper_class_acc: {}'.format(test_loss.avg, acc, per_classes_acc))
        else:
            return acc, per_classes_acc, test_loss.avg
    else:
        acc = torch.sum(torch.squeeze(all_pred).float() == all_label) / float(all_label.size()[0]) * 100
        if args.clip:
            print('CLIP: test_loss {:4f}, test_acc: {:.4f}'.format(test_loss.avg, acc))
        else:
            return acc, test_loss.avg

def obtain_label(model, loader, e, args):
    model.eval()
    class_set = []
    if e == 1:
        return [i for i in range(args.num_class)]  # 使用args.num_class而不是硬编码的65

    number_threshold = 14
    classes_num = [0 for _ in range(args.num_class)]  # 使用args.num_class

    with torch.no_grad():
        for data, _ in loader:
            data = data.to(args.device)
            s_output = model.predict(data)
            preds = torch.max(s_output, 1)[1]
            for pred in preds:
                if pred < args.num_class:  # 确保索引在有效范围内
                    classes_num[pred] += 1

    for c, n in enumerate(classes_num):
        if n >= number_threshold:
            class_set.append(c)

    return class_set

def get_lambda2(epoch, step, max_epoch, max_step):
    """Warm-up λ₂ (GRL 系数)"""
    p = (epoch * max_step + step) / (max_epoch * max_step)
    return 2. / (1. + np.exp(-10 * p)) - 1.

def train(source_loader, mid_loader, target_train_loader, target_test_loader, model, optimizer_g, optimizer_d, scheduler, args):
    # ====== 日志配置 ======
    logging.basicConfig(
        filename=os.path.join(args.log_dir, 'training.log'),
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    n_batch = args.n_iter_per_epoch
    model.apply(fix_bn)
    # --- 自动学习率缩放 ---
    base_bs = 8
    scale = args.l_batch_size / base_bs
    for param_group in optimizer_g.param_groups:
        old_lr = param_group['lr']
        param_group['lr'] = old_lr * scale
    logging.info(f"Adjusted learning rate by batch size scale={scale:.2f}")

    iter_source = itertools.cycle(source_loader)
    iter_mid    = itertools.cycle(mid_loader)
    iter_target = itertools.cycle(target_train_loader)
    scaler = GradScaler(enabled=args.use_amp)

    best_acc = 0

    for e in range(1, args.n_epoch + 1):
        torch.cuda.empty_cache()
        model.train()

        use_stage1 = (e <= args.stage1_epochs)

        if args.pda:
            assert args.datasets in ["office_home", "phyto_plankton"]
            label_set = obtain_label(model, target_train_loader, e, args)
        else:
            label_set = None

        train_loss_clf = AverageMeter()
        train_loss_transfer = AverageMeter()
        train_loss_total = AverageMeter()

        max_step = n_batch
        for step in tqdm(range(n_batch), desc=f"Train:[{e}/{args.n_epoch}]"):

            if use_stage1:
                # ===== Stage1: source + mid 分类 =====
                data_source, label_source = next(iter_source)
                data_mid,    label_mid    = next(iter_mid)
                data_source, label_source = data_source.to(args.device), label_source.to(args.device)
                data_mid,    label_mid    = data_mid.to(args.device),    label_mid.to(args.device)

                if args.use_amp:
                    with autocast():
                        total_loss, loss_s_val, loss_m_val = model.forward_stage1(
                            source=data_source, mid=data_mid,
                            source_label=label_source, mid_label=label_mid
                        )
                    optimizer_g.zero_grad()
                    scaler.scale(total_loss).backward()
                    scaler.step(optimizer_g)
                    scaler.update()
                    loss = total_loss
                else:
                    total_loss, loss_s_val, loss_m_val = model.forward_stage1(
                        source=data_source, mid=data_mid,
                        source_label=label_source, mid_label=label_mid
                    )
                    optimizer_g.zero_grad()
                    total_loss.backward()
                    optimizer_g.step()
                    loss = total_loss

                clf_loss = torch.as_tensor(loss_s_val, device=args.device, dtype=loss.dtype)
                transfer_loss = torch.tensor(0.0, device=args.device, dtype=loss.dtype)

            else:
                # ===== Stage2: mid → target 渐进对齐 =====
                data_mid, label_mid = next(iter_mid)
                if args.fixmatch:
                    (tgt_w, tgt_s), _ = next(iter_target)
                    data_target = tgt_w.to(args.device)
                else:
                    data_target, _ = next(iter_target)
                    data_target = data_target.to(args.device)
                data_mid, label_mid = data_mid.to(args.device), label_mid.to(args.device)

                # ---- Step A: 判别器多步更新 ----
                for _ in range(args.d_steps):
                    with torch.no_grad():
                        feat_m = model.base_network.forward_features(data_mid).detach()
                        feat_t = model.base_network.forward_features(data_target).detach()
                    feat_all = torch.cat([feat_m, feat_t], dim=0)
                    domain_labels = torch.cat([
                        torch.zeros(feat_m.size(0)),
                        torch.ones(feat_t.size(0))
                    ]).to(args.device)

                    logits = model.domain_discriminator(feat_all).squeeze()
                    loss_d = F.binary_cross_entropy_with_logits(logits, domain_labels.float())
                    optimizer_d.zero_grad()
                    loss_d.backward()
                    optimizer_d.step()

                # ---- Step B: 主干更新 + λ₂ warm-up ----
                lambda2 = get_lambda2(e, step, args.n_epoch, max_step)

                if args.use_amp:
                    with autocast():
                        clf_loss, transfer_loss, Dm, Dt = model.forward_stage2(
                            mid=data_mid, target=data_target, mid_label=label_mid, lambda2=lambda2
                        )
                        loss = clf_loss + transfer_loss
                    optimizer_g.zero_grad()
                    scaler.scale(loss).backward()
                    scaler.step(optimizer_g)
                    scaler.update()

                    # 在AMP模式下，Dm和Dt可能是半精度的，需要转换为float32
                    if step % 100 == 0:
                        with torch.no_grad():
                            # 将Dm和Dt转换为float32
                            Dm_float = Dm.float()
                            Dt_float = Dt.float()
                            pred_m = (torch.sigmoid(Dm_float) > 0.5).float()
                            pred_t = (torch.sigmoid(Dt_float) > 0.5).float()
                            acc_m = (pred_m == 0).float().mean()  # 中间域应该被预测为0
                            acc_t = (pred_t == 1).float().mean()  # 目标域应该被预测为1
                            logging.info(f"Step {step}: D_acc_mid={acc_m:.4f}, D_acc_tgt={acc_t:.4f}")
                else:
                    clf_loss, transfer_loss, Dm, Dt = model.forward_stage2(
                        mid=data_mid, target=data_target, mid_label=label_mid, lambda2=lambda2
                    )
                    loss = clf_loss + transfer_loss
                    optimizer_g.zero_grad()
                    loss.backward()
                    optimizer_g.step()

                    # 在训练循环中添加
                    if step % 100 == 0:
                        # 计算判别器准确率
                        with torch.no_grad():
                            pred_m = (torch.sigmoid(Dm) > 0.5).float()
                            pred_t = (torch.sigmoid(Dt) > 0.5).float()
                            acc_m = (pred_m == 0).float().mean()  # 中间域应该被预测为0
                            acc_t = (pred_t == 1).float().mean()  # 目标域应该被预测为1
                            logging.info(f"Step {step}: D_acc_mid={acc_m:.4f}, D_acc_tgt={acc_t:.4f}")

            # ===== 损失统计 =====
            train_loss_clf.update(float(clf_loss.detach()))
            train_loss_transfer.update(float(transfer_loss.detach()))
            train_loss_total.update(float(loss.detach()))

        if scheduler is not None:
            scheduler.step()

        info = f"Epoch: [{e:2d}/{args.n_epoch}], cls_loss: {train_loss_clf.avg:.4f}, transfer_loss: {train_loss_transfer.avg:.4f}, total_loss: {train_loss_total.avg:.4f}"
        if not use_stage1:
            info += f', λ2={lambda2:.3f}, Dm_mean: {Dm.mean().item():.3f}, Dt_mean: {Dt.mean().item():.3f}'

        if args.datasets == "visda":
            test_acc, test_per_class_acc, test_loss = test(model, target_test_loader, args)
            info += f", test_loss {test_loss:.4f}, test_acc: {test_acc:.4f} \nper_class_acc: {test_per_class_acc}"
        else:
            test_acc, test_loss = test(model, target_test_loader, args)
            info += f", test_loss {test_loss:.4f}, test_acc: {test_acc:.4f}"

        if args.rst:
            dsp = rst.dsp_calculation(model)
            info += f", dsp: {dsp:.4f}"

        if best_acc < test_acc:
            best_acc = test_acc
            save_model(model, args)

        logging.info(info)
        tqdm.write(info)
        time.sleep(1)
        torch.cuda.empty_cache()

    tqdm.write(f"Transfer result: {best_acc:.4f}")


def main():
    # os.environ["CUDA_VISIBLE_DEVICES"] = "6"
    parser = get_parser()
    args = parser.parse_args()
    setattr(args, "device", torch.device('cuda:1' if torch.cuda.is_available() else 'cpu'))
    set_random_seed(args.seed)
    # source_loader, mid_loader, target_train_loader, target_test_loader, num_class = load_data(args)
    source_loader, mid_loader, target_train_loader, target_test_loader, num_class = build_dataloaders(args)

    setattr(args, "num_class", num_class)
    setattr(args, "max_iter", 10000)
    log_dir = f'log/{args.model_name}/{args.datasets}/{args.src_domain}2{args.tgt_domain}'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    setattr(args, "log_dir", log_dir)
    print(args)
    model = get_model(args)
    model.to(args.device)
    print(model)
    # optimizer = get_optimizer(model, args)
    optimizer_g, optimizer_d = get_optimizers(model, args)
    scheduler = get_lr_scheduler(optimizer_g, args) if args.scheduler else None

    # if args.scheduler:
    #     scheduler = get_lr_scheduler(optimizer,args)
    # else:
    #     scheduler = None
    print(f"Base Network: {args.model_name}")
    print(f"Source Domain: {args.src_domain}")
    print(f"Mid Domain: {args.mid_domain}")
    print(f"Target Domain: {args.tgt_domain}")
    print(f"FixMatch: {args.fixmatch}")
    print(f"Residual Sparse Training: {args.rst}")
    print(f"Using stage1_epochs = {args.stage1_epochs}")  ### MODIFY ### 输出阶段划分
    if args.rst:
        print(f"Residual Sparse Training Threshold: {args.rst_threshold}")
    if args.clip:
        test(model, target_test_loader, args)
    else:
        train(source_loader, mid_loader, target_train_loader, target_test_loader, model, optimizer_g, optimizer_d, scheduler, args)
    

if __name__ == "__main__":
    main()
