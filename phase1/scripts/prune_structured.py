"""Structured (channel) pruning + fine-tuning with TRUE SHAPE REDUCTION + weight transfer.

Primary method for the study:
- Select important output channels by L1-norm of filters
- Rebuild ResNet-18 with reduced channel counts
- Copy surviving channel weights (and BN stats) into the new model
- Fine-tune with fixed budget
"""
import os
import sys
import json
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(__file__))
from config import (
    DATA_DIR, CKPT_DIR, RESULT_DIR, SEED, FT_EPOCHS, BATCH_SIZE,
    FT_LR, MOMENTUM, WEIGHT_DECAY, NUM_THREADS,
    WARMUP, MEASURE_ITERS_BS1, MEASURE_ITERS_BS32,
)
from models import get_resnet18_cifar, BasicBlock
from utils import (
    set_seed, count_parameters, get_model_size_mb,
    evaluate_accuracy, measure_latency, fix_threads
)
LR = FT_LR
PRUNE_AMOUNTS = [0.4, 0.6]


def get_dataloaders():
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    trainset = torchvision.datasets.CIFAR10(root=DATA_DIR, train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root=DATA_DIR, train=False, download=True, transform=transform_test)
    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=0)
    return trainloader, testloader


def scale(c, amount):
    return max(8, int(round(c * (1.0 - amount))))


class ScaledResNet(nn.Module):
    """ResNet-18 with reduced channel widths. Architecture matches CIFAR style."""
    def __init__(self, amount, num_classes=10):
        super().__init__()
        self.amount = amount
        c64  = scale(64, amount)
        c128 = scale(128, amount)
        c256 = scale(256, amount)
        c512 = scale(512, amount)
        self.channel_config = [c64, c128, c256, c512]

        self.in_planes = c64
        self.conv1 = nn.Conv2d(3, c64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c64)
        self.layer1 = self._make_layer(BasicBlock, c64, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, c128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, c256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, c512, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(c512, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def select_channels(weight, keep):
    """L1-norm importance, return indices of top-keep channels (output dim)."""
    # weight: (out, in, kH, kW)
    importance = weight.data.abs().sum(dim=(1, 2, 3))
    _, idx = torch.topk(importance, keep, largest=True, sorted=False)
    return torch.sort(idx)[0]  # sorted for deterministic transfer


def transfer_conv(src_conv, dst_conv, out_idx, in_idx=None):
    """Copy selected filters. out_idx: which output channels to keep.
       in_idx: which input channels to keep (None = all / identity)."""
    with torch.no_grad():
        if in_idx is None:
            dst_conv.weight.copy_(src_conv.weight.data[out_idx])
        else:
            # both in and out selected
            w = src_conv.weight.data[out_idx][:, in_idx]
            dst_conv.weight.copy_(w)
        if src_conv.bias is not None and dst_conv.bias is not None:
            dst_conv.bias.copy_(src_conv.bias.data[out_idx])


def transfer_bn(src_bn, dst_bn, idx):
    with torch.no_grad():
        dst_bn.weight.copy_(src_bn.weight.data[idx])
        dst_bn.bias.copy_(src_bn.bias.data[idx])
        dst_bn.running_mean.copy_(src_bn.running_mean.data[idx])
        dst_bn.running_var.copy_(src_bn.running_var.data[idx])
        if hasattr(src_bn, 'num_batches_tracked'):
            dst_bn.num_batches_tracked.copy_(src_bn.num_batches_tracked)


def transfer_linear(src, dst, in_idx):
    with torch.no_grad():
        dst.weight.copy_(src.weight.data[:, in_idx])
        if src.bias is not None:
            dst.bias.copy_(src.bias.data)


def build_and_transfer(base_model, amount, device):
    """
    1. Create reduced architecture
    2. For each layer, select important channels by L1
    3. Transfer weights of surviving channels (and adjust next-layer input dims)
    Returns new model with transferred weights.
    """
    new_model = ScaledResNet(amount).to(device)
    base_model = base_model.to(device)
    base_model.eval()

    # Channel counts
    orig = [64, 128, 256, 512]
    keep = [scale(c, amount) for c in orig]

    # ---- conv1 + bn1 ----
    # Input is always 3, select output channels
    out_idx0 = select_channels(base_model.conv1.weight, keep[0])
    transfer_conv(base_model.conv1, new_model.conv1, out_idx0)
    transfer_bn(base_model.bn1, new_model.bn1, out_idx0)
    prev_out_idx = out_idx0

    def transfer_block(src_block, dst_block, in_idx, out_keep):
        """Transfer one BasicBlock. Returns the out_idx used for this block."""
        # conv1 of block: in_channels selected by in_idx, out selected by importance
        out_idx1 = select_channels(src_block.conv1.weight, out_keep)
        transfer_conv(src_block.conv1, dst_block.conv1, out_idx1, in_idx=in_idx)
        transfer_bn(src_block.bn1, dst_block.bn1, out_idx1)

        # conv2 of block: in = out of conv1, out selected
        out_idx2 = select_channels(src_block.conv2.weight, out_keep)
        transfer_conv(src_block.conv2, dst_block.conv2, out_idx2, in_idx=out_idx1)
        transfer_bn(src_block.bn2, dst_block.bn2, out_idx2)

        # shortcut
        if len(src_block.shortcut) > 0:
            # 1x1 conv + bn
            sc_conv = src_block.shortcut[0]
            sc_bn = src_block.shortcut[1]
            # out channels of shortcut should match out of conv2
            # we force the same out_idx2 for residual addition
            transfer_conv(sc_conv, dst_block.shortcut[0], out_idx2, in_idx=in_idx)
            transfer_bn(sc_bn, dst_block.shortcut[1], out_idx2)
        # else identity: residual will be added only if shapes match (they will by construction)

        return out_idx2

    # layer1 (2 blocks), keep[0]
    prev = transfer_block(base_model.layer1[0], new_model.layer1[0], prev_out_idx, keep[0])
    prev = transfer_block(base_model.layer1[1], new_model.layer1[1], prev, keep[0])

    # layer2 (2 blocks), keep[1], first has downsample
    prev = transfer_block(base_model.layer2[0], new_model.layer2[0], prev, keep[1])
    prev = transfer_block(base_model.layer2[1], new_model.layer2[1], prev, keep[1])

    # layer3
    prev = transfer_block(base_model.layer3[0], new_model.layer3[0], prev, keep[2])
    prev = transfer_block(base_model.layer3[1], new_model.layer3[1], prev, keep[2])

    # layer4
    prev = transfer_block(base_model.layer4[0], new_model.layer4[0], prev, keep[3])
    prev = transfer_block(base_model.layer4[1], new_model.layer4[1], prev, keep[3])

    # linear
    transfer_linear(base_model.linear, new_model.linear, prev)

    # Sanity: shapes
    print(f"  Channel config after prune {int(amount*100)}%: {new_model.channel_config}")
    print(f"  Weight transfer completed (L1 channel selection + copy).")
    return new_model


def fine_tune(model, trainloader, testloader, device, epochs=FT_EPOCHS):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    best_acc = 0.0
    for epoch in range(epochs):
        model.train()
        for inputs, targets in trainloader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        scheduler.step()
        acc = evaluate_accuracy(model, testloader, device)
        if acc > best_acc:
            best_acc = acc
        print(f"  FT Epoch {epoch+1}/{epochs} | Test Acc: {acc:.2f}%")
    return model, best_acc


def run_one(amount, baseline_path, device):
    print(f"\n=== Structured Pruning {int(amount*100)}% (TRUE SHAPE + WEIGHT TRANSFER) ===")
    base = get_resnet18_cifar()
    ckpt = torch.load(baseline_path, map_location=device, weights_only=False)
    base.load_state_dict(ckpt["model_state_dict"])
    base = base.to(device)

    total_b, nz_b = count_parameters(base)
    print(f"Baseline params: {total_b:,}")

    model = build_and_transfer(base, amount, device)
    total, nz = count_parameters(model)
    sparsity = 1.0 - (nz / total) if total > 0 else 0.0
    print(f"After shape reduction: total={total:,} nonzero={nz:,} (sparsity vs original ~{(1-total/total_b)*100:.1f}%)")

    trainloader, testloader = get_dataloaders()
    model, best_acc = fine_tune(model, trainloader, testloader, device)

    total, nz = count_parameters(model)
    size = get_model_size_mb(model, path=os.path.join(CKPT_DIR, f"structured_{int(amount*100)}.pth"))
    lat1 = measure_latency(model, batch_size=1, device=device, warmup=WARMUP, iterations=MEASURE_ITERS_BS1)
    lat32 = measure_latency(model, batch_size=32, device=device, warmup=10, iterations=MEASURE_ITERS_BS32)
    throughput = 32 / (lat32["mean"] / 1000.0)

    result = {
        "run_type": "pilot_feasibility",
        "id": f"S{int(amount*100)}",
        "method": f"Structured {int(amount*100)}% (true shape + weight transfer)",
        "compression": f"{int(amount*100)}%",
        "fine_tuned": True,
        "fine_tuning_epochs": FT_EPOCHS,
        "top1_accuracy": best_acc,
        "total_params": total,
        "nonzero_params": nz,
        "theoretical_sparsity": sparsity,
        "param_reduction_vs_baseline": 1.0 - (total / total_b),
        "dense_file_size_mb": size,
        "latency_bs1_mean_ms": lat1["mean"],
        "latency_bs1_std_ms": lat1["std"],
        "latency_bs1_median_ms": lat1["median"],
        "latency_bs32_mean_ms": lat32["mean"],
        "latency_bs32_std_ms": lat32["std"],
        "latency_bs32_median_ms": lat32.get("median"),
        "throughput_bs32": throughput,
        "true_shape_reduction": True,
        "weight_transfer": True,
        "channel_selection": "L1-norm of filters",
        "note": "Primary structured method. Surviving channels' weights + BN stats transferred. Same FT budget. Median latency is primary metric."
    }
    out_path = os.path.join(RESULT_DIR, f"structured_{int(amount*100)}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved {out_path}")
    print(f"Acc={best_acc:.2f}% | Params={total:,} | Size={size:.2f}MB | Lat1 median={lat1['median']:.2f}ms (mean={lat1['mean']:.2f})")
    return result


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--amount", type=float, default=None,
                        help="Channel prune ratio, e.g. 0.4 or 0.6. If omitted, runs 0.4 then 0.6.")
    args = parser.parse_args()
    if args.amount is not None and not (0.0 < args.amount < 1.0):
        # scale() below floors channel counts at 8 regardless of input, so an
        # out-of-range amount (e.g. 1.5) previously produced a silently
        # degenerate 8-channel network instead of failing loudly.
        parser.error(f"--amount must be in (0, 1), got {args.amount}")

    set_seed(SEED)
    n_threads = fix_threads(NUM_THREADS)
    device = torch.device("cpu")
    print(f"Threads fixed to {n_threads}")
    print("Using TRUE SHAPE REDUCTION + WEIGHT TRANSFER as primary structured method.")
    baseline_path = os.path.join(CKPT_DIR, "baseline_fp32_best.pth")
    if not os.path.exists(baseline_path):
        print("Baseline checkpoint not found. Run: python scripts/train_baseline.py")
        sys.exit(1)  # was a plain `return` -> exit code 0, invisible to run_pilot.py's failure check
    amounts = [args.amount] if args.amount is not None else PRUNE_AMOUNTS
    for amt in amounts:
        run_one(amt, baseline_path, device)


if __name__ == "__main__":
    main()
