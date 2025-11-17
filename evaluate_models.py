# evaluate_models_4.py
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet18
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# ============================================================
# Build encoder (same as training)
# ============================================================

def build_encoder(device):
    enc = resnet18(weights=None)
    enc.fc = nn.Identity()
    return enc.to(device)


# ============================================================
# Linear evaluation
# ============================================================

def linear_eval(encoder, train_set, test_loader, device, epochs=20):
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    train_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=0)

    # get embedding dimension
    with torch.no_grad():
        dim = encoder(torch.randn(2,3,96,96).to(device)).shape[-1]

    clf = nn.Linear(dim, 10).to(device)
    opt = torch.optim.SGD(clf.parameters(), lr=0.1)

    # --- train linear classifier ---
    for _ in range(epochs):
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                h = encoder(x)
            loss = nn.CrossEntropyLoss()(clf(h), y)
            opt.zero_grad()
            loss.backward()
            opt.step()

    # --- evaluate ---
    correct = total = 0
    preds, labels = [], []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            h = encoder(x)
            pred = clf(h).argmax(1).cpu()
            preds.extend(pred.tolist())
            labels.extend(y.tolist())
            correct += (pred == y).sum().item()
            total += y.numel()

    return correct / total, np.array(preds), np.array(labels)


# ============================================================
# Plot utils
# ============================================================

def plot_confusion_matrix(cm, classes, title):
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()


def plot_per_class_multi(models_stats, class_names):
    plt.figure(figsize=(12,6))
    x = np.arange(len(class_names))
    width = 0.18

    for i, (name, stats) in enumerate(models_stats.items()):
        plt.bar(x + i*width, stats["per_class_acc"], width, label=name)

    plt.xticks(x + width, class_names, rotation=45)
    plt.ylabel("Per-class Accuracy")
    plt.title("Per-class Accuracy Comparison (4 Models)")
    plt.legend()
    plt.tight_layout()
    plt.show()


# ============================================================
# Main
# ============================================================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # === 4 models ===
    model_files = [
        ("temp0.1_batch128",       "results/temp0.1_batch128_encoder.pt"),
        ("temp0.2_batch128",       "results/temp0.2_batch128_encoder.pt"),
        ("temp0.1_batch256_ep20",  "results/temp0.1_batch256_ep20_encoder.pt"),
        ("temp0.2_batch256_ep20",  "results/temp0.2_batch256_ep20_encoder.pt")
    ]

    # transform
    tf = transforms.Compose([
        transforms.Resize(96),
        transforms.CenterCrop(96),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
    ])

    # datasets
    train_set = datasets.STL10("./data", split="train", download=True, transform=tf)
    test_set  = datasets.STL10("./data", split="test",  download=True, transform=tf)
    test_loader = DataLoader(test_set, batch_size=128, shuffle=False, num_workers=0)

    class_names = train_set.classes

    all_stats = {}

    # ========================================================
    # Evaluate 4 models
    # ========================================================
    for label, path in model_files:
        print(f"\n=== Evaluating {label} ===")

        enc = build_encoder(device)
        state = torch.load(path, map_location=device)
        enc.load_state_dict(state)

        acc, preds, ys = linear_eval(enc, train_set, test_loader, device, epochs=20)

        # confusion matrix
        cm = np.zeros((10,10), dtype=int)
        for p, y in zip(preds, ys):
            cm[y][p] += 1
        per_class = np.diag(cm) / cm.sum(axis=1)

        all_stats[label] = {
            "acc": acc,
            "confusion": cm,
            "per_class_acc": per_class
        }

        print(f"Accuracy = {acc:.4f}")
        print("Per-class accuracy =", per_class)


    # ========================================================
    # Plot confusion matrices
    # ========================================================
    for label in all_stats:
        plot_confusion_matrix(all_stats[label]["confusion"], class_names,
                              f"Confusion Matrix – {label}")

    # ========================================================
    # Plot per-class comparison (4 models)
    # ========================================================
    plot_per_class_multi(all_stats, class_names)


if __name__ == "__main__":
    main()
