# simclr_stl10_min.py
# Minimal, M3-optimized SimCLR on STL-10 (PyTorch, single file)

import argparse
import platform
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Subset

import torchvision
import torchvision.transforms as T
from torchvision.models import resnet18


# -------------------------------
# 1) Data: STL-10 + SimCLR views
# -------------------------------
class TwoCropsTransform:
    """Create two differently augmented views of the same image."""
    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return q, k


def build_simclr_transforms(img_size=96):
    color_jitter = T.ColorJitter(0.4, 0.4, 0.4, 0.1)
    return T.Compose([
        T.RandomResizedCrop(img_size, scale=(0.2, 1.0)),
        T.RandomHorizontalFlip(),
        T.RandomApply([color_jitter], p=0.8),
        T.RandomGrayscale(p=0.2),
        T.GaussianBlur(kernel_size=9, sigma=(0.1, 2.0)),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])


def build_eval_transforms(img_size=96):
    return T.Compose([
        T.Resize(img_size),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])


# -------------------------------------
# 2) Model: Encoder + Projection Head
# -------------------------------------
class ProjectionMLP(nn.Module):
    def __init__(self, in_dim, hid_dim=512, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid_dim, bias=False),
            nn.BatchNorm1d(hid_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hid_dim, out_dim, bias=True),
        )

    def forward(self, x):
        return self.net(x)


class SimCLR(nn.Module):
    def __init__(self, out_dim=128):
        super().__init__()
        backbone = resnet18(weights=None)
        self.feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.encoder = backbone
        self.projector = ProjectionMLP(self.feature_dim, 512, out_dim)

    def forward(self, x):
        h = self.encoder(x)          # [B, feature_dim]
        z = self.projector(h)        # [B, out_dim]
        z = F.normalize(z, dim=1)    # cosine space
        return h, z


# -----------------------------
# 3) NT-Xent (InfoNCE) Loss
# -----------------------------
class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.2):
        super().__init__()
        self.t = temperature

    def forward(self, z1, z2):
        # z1, z2: [B, D], already L2-normalized
        B, _ = z1.size()
        z = torch.cat([z1, z2], dim=0)  # [2B, D]
        sim = torch.matmul(z, z.t()) / self.t  # [2B, 2B] cosine similarity
        mask = torch.eye(2 * B, dtype=torch.bool, device=z.device)
        sim = sim.masked_fill(mask, -9e15)

        pos = torch.cat([torch.diag(sim, B), torch.diag(sim, -B)], dim=0)  # [2B]
        logsumexp = torch.logsumexp(sim, dim=1)  # [2B]
        loss = -pos + logsumexp
        return loss.mean()


# -----------------------------
# 4) Linear evaluation
# -----------------------------
def linear_eval(encoder, train_loader, val_loader, device, epochs=10, lr=0.1):
    """Freeze encoder, train a linear classifier on STL10 train split."""
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    # Detect feature dim with a dummy forward
    feat_dim = encoder(torch.randn(2, 3, 96, 96, device=device)).shape[-1]

    clf = nn.Linear(feat_dim, 10).to(device)
    opt = torch.optim.SGD(clf.parameters(), lr=lr, momentum=0.9, weight_decay=0.)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    for ep in range(epochs):
        clf.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                h = encoder(x)
            logits = clf(h)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        scheduler.step()

    clf.eval()
    total, correct = 0, 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            h = encoder(x)
            logits = clf(h)
            pred = logits.argmax(1)
            total += x.size(0)
            correct += (pred == y).sum().item()

    return correct / total


# -----------------------------
# 5) Training loop (M3-optimized)
# -----------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=str, default='./data')
    # ↓ 缩短训练时间的默认配置，适合 MacBook Air/Pro M 系列
    p.add_argument('--epochs', type=int, default=10, help="SSL pretrain epochs (default: 10)")
    p.add_argument('--batch-size', type=int, default=64, help="SSL batch size (default: 64)")
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--temperature', type=float, default=0.2)
    p.add_argument('--out-dim', type=int, default=128)
    # Mac 上用少量 worker 即可；Windows 仍然强制 0
    default_workers = 0 if platform.system() == "Windows" else 2
    p.add_argument('--num-workers', type=int, default=default_workers)
    p.add_argument('--eval-epochs', type=int, default=10, help="linear eval epochs (default: 10)")
    # 只用一部分 unlabeled 数据做 SSL，加速训练
    p.add_argument('--unlabeled-fraction', type=float, default=0.25,
                   help="fraction of unlabeled STL-10 used for SSL pretraining (0<fr<=1, default: 0.25)")
    args = p.parse_args()

    # -------- Device selection: MPS -> CUDA -> CPU --------
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        device_type = "mps"
        print("Using Apple Silicon GPU (MPS)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        device_type = "cuda"
        print("Using CUDA GPU")
    else:
        device = torch.device("cpu")
        device_type = "cpu"
        print("Using CPU")
    # AMP 只在 CUDA 上启用，MPS 目前效果一般，保持关闭更稳
    use_amp = (device_type == "cuda")
    print(f"device_type={device_type}, use_amp={use_amp}")

    # -------- Datasets --------
    simclr_tf = TwoCropsTransform(build_simclr_transforms(96))
    eval_tf = build_eval_transforms(96)

    train_ssl = torchvision.datasets.STL10(args.data, split='unlabeled',
                                           download=True, transform=simclr_tf)
    train_lbl = torchvision.datasets.STL10(args.data, split='train',
                                           download=True, transform=eval_tf)
    test_lbl = torchvision.datasets.STL10(args.data, split='test',
                                          download=True, transform=eval_tf)

    # 可选：只用一部分 unlabeled 数据做 SSL 预训练（大幅加速）
    if 0 < args.unlabeled_fraction < 1.0:
        full_len = len(train_ssl)
        n_unlabeled = int(full_len * args.unlabeled_fraction)
        indices = list(range(n_unlabeled))
        train_ssl = Subset(train_ssl, indices)
        print(f"[Subset] Using {n_unlabeled}/{full_len} unlabeled images for SSL pretraining "
              f"({args.unlabeled_fraction*100:.1f}%).")

    pin = device_type in ("cuda", "mps")
    loader_ssl = DataLoader(
        train_ssl,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=pin,
        persistent_workers=False,
    )
    loader_lin = DataLoader(
        train_lbl,
        batch_size=256,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin,
        persistent_workers=False,
    )
    loader_val = DataLoader(
        test_lbl,
        batch_size=256,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin,
        persistent_workers=False,
    )

    # -------- Model + optimizer --------
    model = SimCLR(out_dim=args.out_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler(device_type) if use_amp else None
    criterion = NTXentLoss(temperature=args.temperature)

    # -------- Pretrain (SSL) --------
    model.train()
    for ep in range(1, args.epochs + 1):
        total_loss = 0.0
        n_samples = 0
        for i, ((x1, x2), _) in enumerate(loader_ssl, 1):   # ignore pseudo labels (-1)
            x1 = x1.to(device, non_blocking=True)
            x2 = x2.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            if use_amp:
                with autocast(device_type=device_type):
                    _, z1 = model(x1)
                    _, z2 = model(x2)
                    loss = criterion(z1, z2)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                _, z1 = model(x1)
                _, z2 = model(x2)
                loss = criterion(z1, z2)
                loss.backward()
                optimizer.step()

            bs = x1.size(0)
            total_loss += loss.item() * bs
            n_samples += bs

            # batch-level progress
            if i % 50 == 1 or i == len(loader_ssl):
                print(f"[Pretrain] ep {ep} it {i}/{len(loader_ssl)} loss={loss.item():.4f}")

        scheduler.step()
        avg_loss = total_loss / max(1, n_samples)
        print(f"[Pretrain] Epoch {ep}/{args.epochs}  avg_loss={avg_loss:.4f}")

    # -------- Linear evaluation (freeze encoder) --------
    acc = linear_eval(model.encoder, loader_lin, loader_val, device, epochs=args.eval_epochs)
    print(f"Linear probe Top-1 accuracy on STL10 test: {acc * 100:.2f}%")

    # -------- Save encoder --------
    Path('ckpts').mkdir(exist_ok=True)
    torch.save({'encoder': model.encoder.state_dict()}, 'ckpts/simclr_stl10_encoder_m3.pt')
    print('Saved encoder to ckpts/simclr_stl10_encoder_m3.pt')


if __name__ == '__main__':
    import multiprocessing as mp
    mp.freeze_support()  # Windows friendliness
    main()
