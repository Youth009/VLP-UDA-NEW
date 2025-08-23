import os
import torch
import random
import numpy as np
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode


# =============== DataLoader 稳定工具 ===============

# 避免 “Too many open files / Bus error / shm 不足”等问题的默认策略
mp.set_sharing_strategy("file_system")

def seed_worker(worker_id: int):
    """让每个 worker 拿到不同但可复现的随机种子。"""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def make_loader(
    dataset,
    batch_size: int,
    shuffle: bool,
    drop_last: bool,
    num_workers: int,
    seed: int,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    prefetch_factor: int = 2,
):
    """
    一个尽量稳的 DataLoader 封装：
    - persistent_workers：避免频繁重建 worker
    - pin_memory：GPU 训练建议开启
    - prefetch_factor：>0 且 num_workers>0 时有效
    - worker_init_fn + generator：保证复现 & 不同 worker 子种子
    """
    # DataLoader 的随机数生成器（决定 shuffle）
    g = torch.Generator()
    g.manual_seed(seed)

    # prefetch_factor 仅当 num_workers>0 时有效；否则传 None 避免报错
    pf = prefetch_factor if (num_workers and num_workers > 0) else None
    pw = bool(num_workers and num_workers > 0) and persistent_workers

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=pw,
        prefetch_factor=pf,
        worker_init_fn=seed_worker if (num_workers and num_workers > 0) else None,
        generator=g,
    )

class TwoCropsTransform:
    """给 FixMatch 用：返回 (weak, strong) 两个视图。"""
    def __init__(self, weak, strong):
        self.weak = weak
        self.strong = strong

    def __call__(self, x):
        return self.weak(x), self.strong(x)

def build_dataloaders(args):
    """
    构建：
      - source_loader（有标签）
      - mid_loader   （有标签）
      - target_train_loader（无标签/会忽略标签；可 FixMatch）
      - target_test_loader （评测）
    并返回 num_class（以 source 为准）
    """
    # ==== 统一的图像大小 & 归一化（CLIP RN50 推荐的 mean/std）====
    IM_SIZE = 224
    RESIZE_SIZE = 256
    MEAN = (0.48145466, 0.4578275, 0.40821073)
    STD  = (0.26862954, 0.26130258, 0.27577711)

    # 训练弱增强（普适）
    train_weak = transforms.Compose([
        transforms.Resize(RESIZE_SIZE, interpolation=InterpolationMode.BICUBIC),
        transforms.RandomResizedCrop(IM_SIZE, scale=(0.8, 1.0), interpolation=InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    # 训练强增强（如果要用 FixMatch）
    train_strong = transforms.Compose([
        transforms.Resize(RESIZE_SIZE, interpolation=InterpolationMode.BICUBIC),
        transforms.RandomResizedCrop(IM_SIZE, scale=(0.8, 1.0), interpolation=InterpolationMode.BICUBIC),
        transforms.RandomApply([transforms.ColorJitter(.4, .4, .4, .1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    # 测试增强（确定性）
    test_tf = transforms.Compose([
        transforms.Resize(RESIZE_SIZE, interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(IM_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    # ==== 数据集根目录 ====
    src_root = os.path.join(args.data_dir, args.src_domain)
    mid_root = os.path.join(args.data_dir, args.mid_domain)
    tgt_root = os.path.join(args.data_dir, args.tgt_domain)

    # ==== Source & Mid：有标签，直接 ImageFolder ====
    src_ds = datasets.ImageFolder(src_root, transform=train_weak)
    mid_ds = datasets.ImageFolder(mid_root, transform=train_weak)
    num_class = len(src_ds.classes)

    # ==== Target：可能没有 train/test 子目录，做兼容 ====
    tgt_train_dir = os.path.join(tgt_root, "train")
    tgt_test_dir  = os.path.join(tgt_root, "test")

    if os.path.isdir(tgt_train_dir) and os.path.isdir(tgt_test_dir):
        # 标准结构：.../target/train, .../target/test
        if args.fixmatch:
            tgt_train_tf = TwoCropsTransform(train_weak, train_strong)
        else:
            tgt_train_tf = train_weak
        tgt_train_ds = datasets.ImageFolder(tgt_train_dir, transform=tgt_train_tf)
        tgt_test_ds  = datasets.ImageFolder(tgt_test_dir,  transform=test_tf)
    else:
        # 不存在 train/test，就用一个文件夹；再随机划分或“同源评测”
        if args.fixmatch:
            tgt_train_tf = TwoCropsTransform(train_weak, train_strong)
        else:
            tgt_train_tf = train_weak
        tgt_all = datasets.ImageFolder(tgt_root, transform=tgt_train_tf)
        # 这里采用 80/20 划分，你也可以改成“同源评测”（同一个 ds，不分割）
        n_all = len(tgt_all)
        n_train = int(0.8 * n_all)
        n_test  = n_all - n_train
        tgt_train_ds, tgt_test_ds = random_split(
            tgt_all, [n_train, n_test],
            generator=torch.Generator().manual_seed(args.seed)
        )
        # 注意：random_split 返回 Subset；test 需要 test_tf，再包一层
        # 若嫌麻烦可直接把 test 也用弱增强；这里演示换成 test_tf：
        if hasattr(tgt_test_ds, 'dataset'):
            # 替换 Subset 的 transform 为 test_tf
            if isinstance(tgt_test_ds.dataset, datasets.ImageFolder):
                tgt_test_ds.dataset.transform = test_tf

    # ==== 构建稳定的 DataLoader ====
    source_loader = make_loader(
        dataset=src_ds,
        batch_size=args.l_batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        seed=args.seed
    )
    mid_loader = make_loader(
        dataset=mid_ds,
        batch_size=args.l_batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        seed=args.seed
    )
    target_train_loader = make_loader(
        dataset=tgt_train_ds,
        batch_size=args.u_batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        seed=args.seed
    )
    target_test_loader = make_loader(
        dataset=tgt_test_ds,
        batch_size=args.u_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        seed=args.seed
    )

    return source_loader, mid_loader, target_train_loader, target_test_loader, num_class
