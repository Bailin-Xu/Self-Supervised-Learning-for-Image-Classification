# simclr_runAll.py (Improved Version for Assignments)
# Works on Windows + single GPU
# Focus: faster experiments, decent accuracy, richer logs

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms
from torchvision.models import resnet18
from tqdm import tqdm
import os, json, datetime

# ===============================================================
# Utilities
# ===============================================================

def timestamp():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def write_log(path, msg):
    with open(path, "a") as f:
        f.write(f"[{timestamp()}] {msg}\n")
    print(msg)

# ===============================================================
# Transform definitions
# ===============================================================

class TwoCropsTransform:
    def __init__(self, t):
        self.t = t
    def __call__(self, img):
        return self.t(img), self.t(img)

def build_simclr_transform(img=96):
    return transforms.Compose([
        transforms.RandomResizedCrop(img, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.4,0.4,0.4,0.1),
        transforms.RandomGrayscale(0.2),
        transforms.GaussianBlur(kernel_size=9, sigma=(0.1,2.0)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])

def build_eval_transform(img=96):
    return transforms.Compose([
        transforms.Resize(img),
        transforms.CenterCrop(img),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])

# ===============================================================
# Model definitions
# ===============================================================

class ProjectionMLP(nn.Module):
    def __init__(self, in_dim, hid=512, out=128):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dim, hid, bias=False),
            nn.BatchNorm1d(hid),
            nn.ReLU(inplace=True),
            nn.Linear(hid, out)
        )
    def forward(self, x):
        return F.normalize(self.layers(x), dim=1)

class SimCLR(nn.Module):
    def __init__(self, out_dim=128):
        super().__init__()
        base = resnet18(weights=None)
        dim = base.fc.in_features
        base.fc = nn.Identity()
        self.encoder = base
        self.projector = ProjectionMLP(dim, 512, out_dim)
    def forward(self, x):
        return self.projector(self.encoder(x))

class NTXentLoss(nn.Module):
    def __init__(self, t=0.2):
        super().__init__()
        self.t = t
    def forward(self, z1, z2):
        B = z1.size(0)
        z = torch.cat([z1, z2], 0)
        sim = torch.matmul(z, z.T) / self.t
        mask = torch.eye(2*B, device=z.device, dtype=torch.bool)
        sim = sim.masked_fill(mask, -1e9)
        pos = torch.cat([torch.diag(sim, B), torch.diag(sim, -B)])
        loss = -pos + torch.logsumexp(sim, 1)
        return loss.mean()

# ===============================================================
# Linear evaluation with train/val split
# ===============================================================

def linear_eval(encoder, full_train_set, test_loader, device, epochs=50):

    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    # Split train -> train(4000) + val(1000)
    tr_set, val_set = random_split(full_train_set, [4000, 1000])
    tr_loader = DataLoader(tr_set, batch_size=128, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=128, shuffle=False, num_workers=0)

    # dimension
    with torch.no_grad():
        dim = encoder(torch.randn(2,3,96,96, device=device)).shape[-1]

    clf = nn.Linear(dim, 10).to(device)
    opt = torch.optim.SGD(clf.parameters(), lr=0.1)

    train_acc_hist = []
    val_acc_hist = []

    for ep in range(epochs):
        clf.train()
        for x, y in tr_loader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                h = encoder(x)
            loss = F.cross_entropy(clf(h), y)
            opt.zero_grad()
            loss.backward()
            opt.step()

        # --- train acc ---
        clf.eval()
        correct = total = 0
        for x, y in tr_loader:
            x, y = x.to(device), y.to(device)
            pred = clf(encoder(x)).argmax(1)
            correct += (pred == y).sum().item()
            total += y.numel()
        train_acc = correct / total

        # --- val acc ---
        correct = total = 0
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            pred = clf(encoder(x)).argmax(1)
            correct += (pred == y).sum().item()
            total += y.numel()
        val_acc = correct / total

        train_acc_hist.append(train_acc)
        val_acc_hist.append(val_acc)

    # test accuracy
    correct = total = 0
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        pred = clf(encoder(x)).argmax(1)
        correct += (pred == y).sum().item()
        total += y.numel()
    test_acc = correct / total

    return train_acc_hist, val_acc_hist, test_acc

# ===============================================================
# Main Training Loop
# ===============================================================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("results", exist_ok=True)
    log = "results/exp_log.txt"
    write_log(log, f"Start sweep on {device}")

    # hyperparams to sweep
    temps = [0.1, 0.2]  # keep small for homework
    batches = [128]     # 4060 speed-friendly

    E_SSL = 50          # pretrain epochs
    E_LIN = 50          # linear eval epochs

    # data
    ssl_tf = TwoCropsTransform(build_simclr_transform())
    eval_tf = build_eval_transform()

    unlabeled = datasets.STL10("./data", split="unlabeled", download=True, transform=ssl_tf)
    # use 10k subset to save time
    unlabeled = Subset(unlabeled, range(10000))

    full_train = datasets.STL10("./data", split="train", download=True, transform=eval_tf)
    test_set = datasets.STL10("./data", split="test", download=True, transform=eval_tf)

    results = []

    for t in temps:
        for b in batches:
            name = f"temp{t}_batch{b}"
            write_log(log, f"\nRunning {name}")

            loader_ssl = DataLoader(unlabeled, batch_size=b, shuffle=True, drop_last=True, num_workers=0)
            loader_test = DataLoader(test_set, batch_size=128, shuffle=False, num_workers=0)

            model = SimCLR(out_dim=128).to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
            criterion = NTXentLoss(t)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=E_SSL)

            ssl_loss_hist = []

            # =======================
            # SSL Pretraining
            # =======================
            for ep in range(1, E_SSL+1):
                model.train()
                loss_sum = 0

                for (x1, x2), _ in tqdm(loader_ssl, desc=f"{name} SSL Ep{ep}/{E_SSL}"):
                    x1, x2 = x1.to(device), x2.to(device)
                    z1 = model(x1)
                    z2 = model(x2)
                    loss = criterion(z1, z2)

                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    loss_sum += loss.item()

                scheduler.step()
                avg_loss = loss_sum / len(loader_ssl)
                ssl_loss_hist.append(avg_loss)
                write_log(log, f"{name} Ep{ep} SSL-Loss={avg_loss:.4f}")

            # Save encoder weights
            torch.save(model.encoder.state_dict(), f"results/{name}_encoder.pt")

            # =======================
            # Linear eval
            # =======================
            tr_hist, val_hist, test_acc = linear_eval(model.encoder, full_train, loader_test, device, epochs=E_LIN)
            write_log(log, f"{name} TestAcc={test_acc:.4f}")

            results.append({
                "name": name,
                "temp": t,
                "batch": b,
                "ssl_loss": ssl_loss_hist,
                "train_acc_curve": tr_hist,
                "val_acc_curve": val_hist,
                "test_acc": test_acc
            })

    with open("results/summary.json","w") as f:
        json.dump(results, f, indent=4)

    write_log(log, "Sweep Completed.")

if __name__ == "__main__":
    main()
