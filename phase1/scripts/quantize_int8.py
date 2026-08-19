"""INT8 Post-Training Quantization. Separate branch from pruning. No fine-tuning."""
import os
import sys
import json
import platform
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(__file__))
from config import (
    DATA_DIR, CKPT_DIR, RESULT_DIR, SEED, NUM_THREADS,
    WARMUP, MEASURE_ITERS_BS1, MEASURE_ITERS_BS32,
)
from models import get_resnet18_cifar
from utils import (
    set_seed, count_parameters, evaluate_accuracy, measure_latency, fix_threads,
)


def get_dataloaders():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    trainset = torchvision.datasets.CIFAR10(root=DATA_DIR, train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root=DATA_DIR, train=False, download=True, transform=transform)
    calib_loader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=0)
    testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=0)
    return calib_loader, testloader


def select_quant_backend():
    """
    x86/amd64 -> fbgemm
    Apple Silicon / ARM -> qnnpack
    Never force fbgemm on arm64.
    """
    machine = platform.machine().lower()
    supported = []
    try:
        supported = list(torch.backends.quantized.supported_engines)
    except Exception:
        pass

    if machine in ("arm64", "aarch64"):
        preferred = "qnnpack"
    else:
        preferred = "fbgemm"

    if preferred in supported:
        engine = preferred
    elif "qnnpack" in supported:
        engine = "qnnpack"
    elif "fbgemm" in supported:
        engine = "fbgemm"
    elif supported:
        engine = supported[0]
    else:
        engine = preferred

    torch.backends.quantized.engine = engine
    print(f"CPU arch: {machine}")
    print(f"Supported quantized engines: {supported}")
    print(f"Using quantized engine: {engine}")
    return engine


def quantize_model(model, calib_loader, device, engine, num_calib_batches=20):
    model.eval()
    model.to(device)
    model.qconfig = torch.quantization.get_default_qconfig(engine)
    model_prepared = torch.quantization.prepare(model, inplace=False)
    print("Calibrating...")
    with torch.no_grad():
        for i, (inputs, _) in enumerate(calib_loader):
            if i >= num_calib_batches:
                break
            _ = model_prepared(inputs.to(device))
    return torch.quantization.convert(model_prepared, inplace=False)


def main():
    set_seed(SEED)
    n_threads = fix_threads(NUM_THREADS)
    device = torch.device("cpu")
    print("=" * 60)
    print("PILOT — INT8 PTQ (separate branch, no fine-tuning)")
    print("=" * 60)
    print(f"Device: {device} | Threads: {n_threads}")
    engine = select_quant_backend()

    baseline_path = os.path.join(CKPT_DIR, "baseline_fp32_best.pth")
    if not os.path.exists(baseline_path):
        print("Baseline not found. Run: python scripts/train_baseline.py")
        sys.exit(1)  # was a plain `return` -> exit code 0, invisible to run_pilot.py's failure check

    model = get_resnet18_cifar()
    ckpt = torch.load(baseline_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    calib_loader, testloader = get_dataloaders()
    fp32_acc = evaluate_accuracy(model, testloader, device)
    print(f"FP32 reference Acc: {fp32_acc:.2f}%")

    try:
        model_int8 = quantize_model(model, calib_loader, device, engine)
        int8_acc = evaluate_accuracy(model_int8, testloader, device)
        quant_type = "static_ptq"
        print(f"INT8 (static) Acc: {int8_acc:.2f}%")
    except Exception as e:
        print(f"Static quant failed: {e}. Falling back to dynamic.")
        model_int8 = torch.quantization.quantize_dynamic(
            model, {nn.Linear, nn.Conv2d}, dtype=torch.qint8
        )
        int8_acc = evaluate_accuracy(model_int8, testloader, device)
        quant_type = "dynamic"
        print(f"Dynamic INT8 Acc: {int8_acc:.2f}%")

    total, nz = count_parameters(model_int8)
    size_path = os.path.join(CKPT_DIR, "int8_model.pth")
    torch.save(model_int8.state_dict(), size_path)
    size = os.path.getsize(size_path) / (1024 * 1024)

    lat1 = measure_latency(model_int8, batch_size=1, device=device, warmup=WARMUP, iterations=MEASURE_ITERS_BS1)
    lat32 = measure_latency(model_int8, batch_size=32, device=device, warmup=10, iterations=MEASURE_ITERS_BS32)
    throughput = 32 / (lat32["mean"] / 1000.0)

    result = {
        "run_type": "pilot_feasibility",
        "id": "Q",
        "method": "INT8 Quantization (PTQ)",
        "compression": "INT8",
        "fine_tuned": False,
        "fine_tuning_epochs": 0,
        "quant_type": quant_type,
        "top1_accuracy": int8_acc,
        "fp32_reference_acc": fp32_acc,
        "total_params": total,
        "nonzero_params": nz,
        "dense_file_size_mb": size,
        "latency_bs1_mean_ms": lat1["mean"],
        "latency_bs1_std_ms": lat1["std"],
        "latency_bs1_median_ms": lat1["median"],
        "latency_bs32_mean_ms": lat32["mean"],
        "latency_bs32_std_ms": lat32["std"],
        "latency_bs32_median_ms": lat32.get("median"),
        "throughput_bs32": throughput,
        "backend": engine,
        "cpu_arch": platform.machine(),
        "note": "Separate branch from pruning. CPU INT8 PTQ, no FT. Backend is fbgemm on x86 and qnnpack on Apple Silicon/ARM. Median latency is primary.",
    }
    with open(os.path.join(RESULT_DIR, "int8_results.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved. Acc={int8_acc:.2f}% | Size={size:.2f}MB | "
          f"Lat1 median={lat1['median']:.2f}ms (mean={lat1['mean']:.2f})")


if __name__ == "__main__":
    main()
