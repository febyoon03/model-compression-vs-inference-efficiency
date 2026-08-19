"""Step 1: Train FP32 ResNet-18 baseline on CIFAR-10 and measure metrics."""
import os
import sys
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(__file__))
from config import (
    DATA_DIR, CKPT_DIR, RESULT_DIR, SEED, EPOCHS, BATCH_SIZE,
    LR, MOMENTUM, WEIGHT_DECAY, NUM_WORKERS, NUM_THREADS,
    WARMUP, MEASURE_ITERS_BS1, MEASURE_ITERS_BS32,
)
from models import get_resnet18_cifar
from utils import (
    set_seed, count_parameters, get_model_size_mb,
    evaluate_accuracy, measure_latency, measure_peak_memory, fix_threads,
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
    trainset = torchvision.datasets.CIFAR10(
        root=DATA_DIR, train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(
        root=DATA_DIR, train=False, download=True, transform=transform_test)
    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True,
                             num_workers=NUM_WORKERS, pin_memory=False)
    testloader = DataLoader(testset, batch_size=100, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=False)
    return trainloader, testloader


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    return running_loss / len(loader), 100. * correct / total


def main():
    set_seed(SEED)
    n_threads = fix_threads(NUM_THREADS)
    device = torch.device("cpu")
    print("=" * 60)
    print("PILOT / FEASIBILITY RUN — Baseline FP32")
    print("Accuracy is for pipeline validation only. Not final research result.")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Torch threads fixed to: {n_threads}")

    trainloader, testloader = get_dataloaders()
    print(f"Train samples: {len(trainloader.dataset)}, Test: {len(testloader.dataset)}")

    model = get_resnet18_cifar(num_classes=10).to(device)
    print(f"Model created. Params: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_acc = 0.0
    history = []
    print(f"\n=== Training baseline for {EPOCHS} epochs ===")
    start_time = time.time()
    for epoch in range(EPOCHS):
        train_loss, train_acc = train_one_epoch(model, trainloader, optimizer, criterion, device)
        test_acc = evaluate_accuracy(model, testloader, device)
        scheduler.step()
        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "test_acc": test_acc,
            "lr": optimizer.param_groups[0]["lr"],
        })
        print(f"Epoch {epoch+1:3d}/{EPOCHS} | Loss: {train_loss:.4f} | "
              f"Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%")
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "acc": best_acc,
            }, os.path.join(CKPT_DIR, "baseline_fp32_best.pth"))

    total_time = time.time() - start_time
    print(f"\nTraining finished in {total_time/60:.1f} min. Best Acc: {best_acc:.2f}%")

    ckpt = torch.load(os.path.join(CKPT_DIR, "baseline_fp32_best.pth"),
                      map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print("\n=== Measuring baseline metrics ===")
    total_params, nonzero_params = count_parameters(model)
    dense_size = get_model_size_mb(model, path=os.path.join(CKPT_DIR, "baseline_fp32_state.pth"))
    final_acc = evaluate_accuracy(model, testloader, device)

    lat_bs1 = measure_latency(model, batch_size=1, device=device,
                              warmup=WARMUP, iterations=MEASURE_ITERS_BS1)
    lat_bs32 = measure_latency(model, batch_size=32, device=device,
                               warmup=10, iterations=MEASURE_ITERS_BS32)
    throughput = 32 / (lat_bs32["mean"] / 1000.0)

    try:
        mem = measure_peak_memory(model, batch_size=1, device=device)
    except Exception as e:
        mem = None
        print(f"Memory measure failed: {e}")

    results = {
        "run_type": "pilot_feasibility",
        "method": "FP32 Baseline",
        "id": "B",
        "compression": "0%",
        "fine_tuned": False,
        "fine_tuning_epochs": 0,
        "top1_accuracy": final_acc,
        "total_params": total_params,
        "nonzero_params": nonzero_params,
        "theoretical_sparsity": 0.0,
        "dense_file_size_mb": dense_size,
        "latency_bs1_mean_ms": lat_bs1["mean"],
        "latency_bs1_std_ms": lat_bs1["std"],
        "latency_bs1_median_ms": lat_bs1["median"],
        "latency_bs32_mean_ms": lat_bs32["mean"],
        "latency_bs32_std_ms": lat_bs32["std"],
        "latency_bs32_median_ms": lat_bs32.get("median"),
        "throughput_bs32": throughput,
        "peak_memory_mb": mem,
        "training_epochs": EPOCHS,
        "training_time_min": total_time / 60,
        "history": history,
        "hardware": {
            "device": str(device),
            "cpu_threads": n_threads,
            "torch_version": torch.__version__,
            "note": "CPU inference focus",
        },
        "note": "Pilot run. Accuracy not to be interpreted as final research result. Median latency is primary.",
    }
    with open(os.path.join(RESULT_DIR, "baseline_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print("\n===== BASELINE RESULTS =====")
    print(f"Top-1 Accuracy      : {final_acc:.2f}%")
    print(f"Total Parameters    : {total_params:,}")
    print(f"Non-zero Parameters : {nonzero_params:,}")
    print(f"Dense File Size     : {dense_size:.2f} MB")
    print(f"Latency BS=1 median : {lat_bs1['median']:.2f} ms  (mean {lat_bs1['mean']:.2f} ± {lat_bs1['std']:.2f})")
    print(f"Latency BS=32       : {lat_bs32['mean']:.2f} ± {lat_bs32['std']:.2f} ms")
    print(f"Throughput BS=32    : {throughput:.1f} img/s")
    print(f"Saved: {RESULT_DIR}/baseline_results.json")
    print("============================")


if __name__ == "__main__":
    main()
