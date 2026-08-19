"""Unstructured magnitude pruning + fine-tuning. Dense tensor shapes unchanged."""
import argparse
import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.nn.utils import prune

sys.path.append(os.path.dirname(__file__))
from config import (
    DATA_DIR, CKPT_DIR, RESULT_DIR, SEED, FT_EPOCHS, BATCH_SIZE,
    FT_LR, MOMENTUM, WEIGHT_DECAY, NUM_THREADS,
    WARMUP, MEASURE_ITERS_BS1, MEASURE_ITERS_BS32,
)
from models import get_resnet18_cifar
from utils import (
    set_seed, count_parameters, get_model_size_mb,
    evaluate_accuracy, measure_latency, fix_threads,
)


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


def apply_unstructured_pruning(model, amount):
    """Global unstructured L1 pruning. Keep prune hooks active during FT."""
    parameters_to_prune = []
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            parameters_to_prune.append((module, "weight"))
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=amount,
    )
    # Do NOT prune.remove() here. Mask must stay during fine-tuning.
    return model, parameters_to_prune


def make_pruning_permanent(parameters_to_prune):
    """Bake mask into weight tensors after fine-tuning is finished."""
    for module, param_name in parameters_to_prune:
        prune.remove(module, param_name)


def count_effective_conv_linear_weights(model):
    """Count nonzero in the *effective* (masked) Conv/Linear weights."""
    total = 0
    nonzero = 0
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            w = module.weight.detach()
            total += w.numel()
            nonzero += (w != 0).sum().item()
    return total, nonzero


def fine_tune(model, trainloader, testloader, device, epochs=FT_EPOCHS):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=FT_LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
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
        w_total, w_nz = count_effective_conv_linear_weights(model)
        print(f"  FT Epoch {epoch+1}/{epochs} | Test Acc: {acc:.2f}% | "
              f"weight sparsity={(1 - w_nz / w_total) * 100:.1f}%")
    return model, best_acc


def run_one(amount, baseline_path, device):
    print(f"\n=== Unstructured Pruning {int(amount*100)}% (Pilot) ===")
    print("Note: tensor shapes unchanged. Zeros stay in dense tensors.")
    model = get_resnet18_cifar()
    ckpt = torch.load(baseline_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)

    total_before, nz_before = count_parameters(model)
    print(f"Before prune: total={total_before:,} nonzero={nz_before:,}")

    model, pruned_params = apply_unstructured_pruning(model, amount)
    w_total, w_nz = count_effective_conv_linear_weights(model)
    sparsity_after_prune = 1.0 - (w_nz / w_total)
    print(f"After prune (mask on): conv/linear weights nonzero={w_nz:,}/{w_total:,} "
          f"sparsity={sparsity_after_prune*100:.1f}%")

    trainloader, testloader = get_dataloaders()
    model, best_acc = fine_tune(model, trainloader, testloader, device)

    # Bake zeros into the tensors only after FT, then measure
    make_pruning_permanent(pruned_params)
    total, nz = count_parameters(model)
    sparsity = 1.0 - (nz / total) if total > 0 else 0.0
    print(f"After FT + remove(): total={total:,} nonzero={nz:,} sparsity={sparsity*100:.1f}%")

    size = get_model_size_mb(model, path=os.path.join(CKPT_DIR, f"unstructured_{int(amount*100)}.pth"))
    lat1 = measure_latency(model, batch_size=1, device=device, warmup=WARMUP, iterations=MEASURE_ITERS_BS1)
    lat32 = measure_latency(model, batch_size=32, device=device, warmup=10, iterations=MEASURE_ITERS_BS32)
    throughput = 32 / (lat32["mean"] / 1000.0)

    result = {
        "run_type": "pilot_feasibility",
        "id": f"U{int(amount*100)}",
        "method": f"Unstructured {int(amount*100)}%",
        "compression": f"{int(amount*100)}%",
        "fine_tuned": True,
        "fine_tuning_epochs": FT_EPOCHS,
        "top1_accuracy": best_acc,
        "total_params": total,
        "nonzero_params": nz,
        "theoretical_sparsity": sparsity,
        "sparsity_after_prune_weights": sparsity_after_prune,
        "dense_file_size_mb": size,
        "latency_bs1_mean_ms": lat1["mean"],
        "latency_bs1_std_ms": lat1["std"],
        "latency_bs1_median_ms": lat1["median"],
        "latency_bs32_mean_ms": lat32["mean"],
        "latency_bs32_std_ms": lat32["std"],
        "latency_bs32_median_ms": lat32.get("median"),
        "throughput_bs32": throughput,
        "note": "Dense tensors retained; zeros present but shape unchanged. Same FT budget. Median latency is primary.",
    }
    out_path = os.path.join(RESULT_DIR, f"unstructured_{int(amount*100)}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved {out_path}")
    print(f"Acc={best_acc:.2f}% | NZ={nz:,} | Size={size:.2f}MB | "
          f"Lat1 median={lat1['median']:.2f}ms (mean={lat1['mean']:.2f})")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--amount", type=float, default=None,
                        help="Prune ratio, e.g. 0.4 or 0.6. If omitted, runs 0.4 then 0.6.")
    args = parser.parse_args()
    if args.amount is not None and not (0.0 < args.amount < 1.0):
        # prune.global_unstructured also rejects this internally, but fail
        # here with a clearer message before any data loading/training runs.
        parser.error(f"--amount must be in (0, 1), got {args.amount}")

    set_seed(SEED)
    n_threads = fix_threads(NUM_THREADS)
    device = torch.device("cpu")
    print(f"Threads fixed to {n_threads}")
    baseline_path = os.path.join(CKPT_DIR, "baseline_fp32_best.pth")
    if not os.path.exists(baseline_path):
        print("Baseline checkpoint not found. Run: python scripts/train_baseline.py")
        sys.exit(1)  # was a plain `return` -> exit code 0, invisible to run_pilot.py's failure check
    amounts = [args.amount] if args.amount is not None else [0.4, 0.6]
    for amt in amounts:
        run_one(amt, baseline_path, device)


if __name__ == "__main__":
    main()
