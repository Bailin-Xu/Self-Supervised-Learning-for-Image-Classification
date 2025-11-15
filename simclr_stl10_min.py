# simclr_stl10_min.py
# Minimal, readable SimCLR on STL-10 (PyTorch, single file)

import argparse
import platform
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader

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
        sim = torch.matmul(z, z.t()) / self.t  # [2B, 2B] cosine similarity (since z normalized)
        mask = torch.eye(2*B, dtype=torch.bool, device=z.device)
        sim = sim.masked_fill(mask, -9e15)
        pos = torch.cat([torch.diag(sim, B), torch.diag(sim, -B)], dim=0)  # [2B]
        logsumexp = torch.logsumexp(sim, dim=1)  # [2B]
        loss = -pos + logsumexp
        return loss.mean()


# -----------------------------
# 4) Utils
# -----------------------------
def linear_eval(encoder, train_loader, val_loader, device, epochs=50, lr=0.1):
    """Freeze encoder, train a linear classifier on STL10 train split."""
    encoder.eval()
    # 显式冻结 encoder 参数（可选但推荐）
    for p in encoder.parameters():
        p.requires_grad = False

    # 用一小批随机张量探测特征维度
    feat_dim = encoder(torch.randn(2, 3, 96, 96, device=device)).shape[-1]

    # 仅训练线性分类头
    clf = nn.Linear(feat_dim, 10).to(device)
    opt = torch.optim.SGD(clf.parameters(), lr=lr, momentum=0.9, weight_decay=0.)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    for _ in range(epochs):
        clf.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            # 冻结 encoder：只在这段禁用梯度
            with torch.no_grad():
                h = encoder(x)
            logits = clf(h)                    # <- 这里需要梯度
            loss = F.cross_entropy(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        scheduler.step()

    # 验证
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
# 5) Training loop
# -----------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=str, default='./data')
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--temperature', type=float, default=0.2)
    p.add_argument('--out-dim', type=int, default=128)
    p.add_argument('--num-workers', type=int, default=0 if platform.system() == "Windows" else 4)
    p.add_argument('--eval-epochs', type=int, default=50)
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
    use_amp = (device_type == 'cuda')  # only use AMP on CUDA
    print('Using device:', device)

    # Datasets
    simclr_tf = TwoCropsTransform(build_simclr_transforms(96))
    eval_tf   = build_eval_transforms(96)

    train_ssl = torchvision.datasets.STL10(args.data, split='unlabeled', download=True, transform=simclr_tf)
    train_lbl = torchvision.datasets.STL10(args.data, split='train',     download=True, transform=eval_tf)
    test_lbl  = torchvision.datasets.STL10(args.data, split='test',      download=True, transform=eval_tf)

    pin = (device_type == 'cuda')
    loader_ssl = DataLoader(train_ssl, batch_size=args.batch_size, shuffle=True, drop_last=True,
                            num_workers=args.num_workers, pin_memory=pin, persistent_workers=False)
    loader_lin = DataLoader(train_lbl, batch_size=256, shuffle=True,
                            num_workers=args.num_workers, pin_memory=pin, persistent_workers=False)
    loader_val = DataLoader(test_lbl,  batch_size=256, shuffle=False,
                            num_workers=args.num_workers, pin_memory=pin, persistent_workers=False)

    # Model + optimizer
    model = SimCLR(out_dim=args.out_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler(device_type) if use_amp else None
    criterion = NTXentLoss(temperature=args.temperature)

    # Pretrain
    model.train()
    for ep in range(1, args.epochs + 1):
        total_loss = 0.0
        for i, ((x1, x2), _) in enumerate(loader_ssl, 1):   # ignore pseudo labels (-1)
            x1 = x1.to(device, non_blocking=True)
            x2 = x2.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=device_type, enabled=use_amp):
                _, z1 = model(x1)
                _, z2 = model(x2)
                loss = criterion(z1, z2)

            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * x1.size(0)

            # batch-level progress (prints every ~50 iters)
            if i % 50 == 1 or i == len(loader_ssl):
                print(f"[Pretrain] ep {ep} it {i}/{len(loader_ssl)} loss={loss.item():.4f}")

        scheduler.step()
        denom = len(loader_ssl.dataset) - (len(loader_ssl.dataset) % args.batch_size)
        avg_loss = total_loss / max(1, denom)
        print(f"[Pretrain] Epoch {ep}/{args.epochs}  loss={avg_loss:.4f}")

    # Linear evaluation (freeze encoder)
    acc = linear_eval(model.encoder, loader_lin, loader_val, device, epochs=args.eval_epochs)
    print(f"Linear probe Top-1 accuracy on STL10 test: {acc*100:.2f}%")

    # Save encoder
    Path('ckpts').mkdir(exist_ok=True)
    torch.save({'encoder': model.encoder.state_dict()}, 'ckpts/simclr_stl10_encoder.pt')
    print('Saved encoder to ckpts/simclr_stl10_encoder.pt')


if __name__ == '__main__':
    import multiprocessing as mp
    mp.freeze_support()  # Windows friendliness
    main()
