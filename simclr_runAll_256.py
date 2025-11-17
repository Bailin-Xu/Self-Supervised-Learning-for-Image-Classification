# simclr_runAll_256_20.py
# Sweep over temp=[0.1,0.2], fixed batch=256, SSL epoch=20

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.models import resnet18
from tqdm import tqdm
import os, json, datetime

def timestamp():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def write_log(path, msg):
    with open(path, "a") as f:
        f.write(f"[{timestamp()}] {msg}\n")
    print(msg)

# ---------------------- Transform ----------------------

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

# ---------------------- Models ----------------------

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
        net = resnet18(weights=None)
        f = net.fc.in_features
        net.fc = nn.Identity()
        self.encoder = net
        self.projector = ProjectionMLP(f, 512, out_dim)

    def forward(self, x):
        return self.projector(self.encoder(x))

# ---------------------- NT-Xent Loss ----------------------

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

# ---------------------- Linear Eval ----------------------

def linear_eval(encoder, tr_loader, te_loader, device):
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    dim = encoder(torch.randn(2,3,96,96, device=device)).shape[-1]
    clf = nn.Linear(dim, 10).to(device)
    opt = torch.optim.SGD(clf.parameters(), lr=0.1)

    for _ in range(20):  # fixed 20 epochs
        for x, y in tr_loader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                h = encoder(x)
            loss = F.cross_entropy(clf(h), y)
            opt.zero_grad()
            loss.backward()
            opt.step()

    correct = total = 0
    with torch.no_grad():
        for x, y in te_loader:
            x, y = x.to(device), y.to(device)
            pred = clf(encoder(x)).argmax(1)
            correct += (pred == y).sum().item()
            total += y.numel()

    return correct / total


# ---------------------- Main ----------------------

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("results", exist_ok=True)
    log_path = "results/exp_log_256_20.txt"

    write_log(log_path, f"Start sweep on {device}")

    temps = [0.1, 0.2]
    batches = [256]          # <--- fixed
    proj_dims = [128]        # same as before
    SSL_EPOCHS = 20          # <--- changed from 50 to 20

    ssl_tf = TwoCropsTransform(build_simclr_transform())
    eval_tf = build_eval_transform()

    # load data
    unlabeled = datasets.STL10("./data", split="unlabeled", download=True, transform=ssl_tf)
    unlabeled = Subset(unlabeled, range(10000))

    train_lbl = datasets.STL10("./data", split="train", download=True, transform=eval_tf)
    test_lbl  = datasets.STL10("./data", split="test", download=True, transform=eval_tf)

    results = []

    for t in temps:
        for b in batches:
            for pd in proj_dims:

                name = f"temp{t}_batch{b}_ep20"
                write_log(log_path, f"Running {name}")

                loader_ssl = DataLoader(unlabeled, batch_size=b, shuffle=True, drop_last=True)
                loader_tr  = DataLoader(train_lbl, batch_size=128, shuffle=True)
                loader_te  = DataLoader(test_lbl, batch_size=128)

                # model
                model = SimCLR(out_dim=pd).to(device)
                opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
                criterion = NTXentLoss(t)

                # SSL training
                model.train()
                for ep in range(1, SSL_EPOCHS + 1):
                    loss_sum = 0
                    for (x1, x2), _ in tqdm(loader_ssl, desc=f"{name} Ep{ep}/{SSL_EPOCHS}"):
                        x1, x2 = x1.to(device), x2.to(device)
                        opt.zero_grad()
                        loss = criterion(model(x1), model(x2))
                        loss.backward()
                        opt.step()
                        loss_sum += loss.item()

                    write_log(log_path, f"{name} Ep{ep} SSL-Loss={loss_sum/len(loader_ssl):.4f}")

                # linear eval
                acc = linear_eval(model.encoder, loader_tr, loader_te, device)
                write_log(log_path, f"{name} Acc={acc:.4f}")

                torch.save(model.encoder.state_dict(), f"results/{name}_encoder.pt")

                results.append({
                    "exp": name,
                    "temp": t,
                    "batch": b,
                    "proj_dim": pd,
                    "epoch": SSL_EPOCHS,
                    "acc": acc
                })

    with open("results/summary_256_20.json", "w") as f:
        json.dump(results, f, indent=4)

    write_log(log_path, "Sweep Completed.")

if __name__ == "__main__":
    main()
