"""
train.py - train ResNet-8 on CIFAR-10 and save its parameters for quantize.py.

    python train.py                      # 60 epochs, downloads CIFAR-10 into ./data
    python train.py --epochs 5           # a quick run

Inputs are scaled exactly like the integer model sees them: (pixel - 128) / 128.
The output resnet8_float.npz holds the PyTorch parameters (batch norm not yet
folded); at the end, the NumPy float model is checked against PyTorch.
"""
import argparse
import time

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T

import cifar10
from graph import resnet8_graph
from nn_numpy import fold_bn, forward_float
from resnet8 import ResNet8


def to_model_range(t):
    """ToTensor gives pixel / 255; the model wants (pixel - 128) / 128."""
    return (t * 255.0 - 128.0) / 128.0


def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            pred = model(x.to(device)).argmax(1).cpu()
            correct += (pred == y).sum().item()
            total += y.numel()
    return 100.0 * correct / total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--wd", type=float, default=5e-4)
    ap.add_argument("--data", default="data")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", default="resnet8_float.npz")
    args = ap.parse_args()

    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    norm = T.Lambda(to_model_range)
    train_set = torchvision.datasets.CIFAR10(args.data, train=True, download=True, transform=T.Compose(
        [T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(), T.ToTensor(), norm]))
    test_set = torchvision.datasets.CIFAR10(args.data, train=False, download=True,
                                            transform=T.Compose([T.ToTensor(), norm]))
    train_loader = torch.utils.data.DataLoader(train_set, args.batch, shuffle=True,
                                               num_workers=args.workers, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_set, 500, num_workers=args.workers)

    model = ResNet8().to(device)
    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,
                          weight_decay=args.wd, nesterov=True)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs * len(train_loader))
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, seen, loss_sum = time.time(), 0, 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            loss = loss_fn(model(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            loss_sum += loss.item() * y.numel()
            seen += y.numel()
        acc = evaluate(model, test_loader, device)
        print(f"epoch {epoch:3d}  loss {loss_sum / seen:.3f}  test {acc:.2f}%  ({time.time() - t0:.0f} s)")

    params = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}
    np.savez(args.out, **params)
    print(f"saved {args.out}")

    # the NumPy float model must agree with PyTorch, or quantize.py would start from a wrong model
    imgs, labels = cifar10.load(args.data, train=False)
    graph = resnet8_graph()
    logits = forward_float(fold_bn(params, graph), graph, cifar10.to_float(imgs[:2000]))["logits"]
    np_acc = 100.0 * (logits.argmax(1) == labels[:2000]).mean()
    model.eval()
    with torch.no_grad():
        x = torch.tensor(cifar10.to_float(imgs[:2000]), dtype=torch.float32, device=device)
        torch_acc = 100.0 * (model(x).argmax(1).cpu().numpy() == labels[:2000]).mean()
    print(f"first 2000 test images: PyTorch {torch_acc:.2f}%, NumPy float model {np_acc:.2f}%"
          + ("" if abs(torch_acc - np_acc) < 0.3 else "   <-- MISMATCH, check nn_numpy.py"))


if __name__ == "__main__":
    main()
