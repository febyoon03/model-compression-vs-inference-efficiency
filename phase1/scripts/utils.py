"""Utility functions for model measurement and evaluation."""
import os
import time
import torch
import torch.nn as nn
from collections import OrderedDict
import numpy as np

# Experimental control: fix CPU threads for all measurements
def fix_threads(n=2):
    torch.set_num_threads(n)
    try:
        torch.set_num_interop_threads(1)
    except Exception:
        pass
    return torch.get_num_threads()


def count_parameters(model):
    """Count total and non-zero parameters.

    Standard nn.Parameter weights (Conv2d, Linear, BatchNorm, ...) are
    counted via model.parameters(). Layers converted by
    torch.quantization.quantize_dynamic (e.g. a quantized Linear) store
    their weight in a packed buffer accessed via a *callable* module.weight()
    rather than a registered nn.Parameter, so model.parameters() silently
    skips them entirely -- undercounting the model by exactly that layer's
    size (verified on this pilot's INT8 run: reported total_params was short
    by 5,130, exactly matching ResNet-18's final Linear(512, 10) layer).
    Those modules are detected here (module.weight is callable, unlike a
    plain nn.Parameter) and added back in via their weight()/bias()
    accessors, dequantized for a like-for-like nonzero comparison.
    """
    total = 0
    nonzero = 0
    for p in model.parameters():
        total += p.numel()
        nonzero += (p != 0).sum().item()

    for module in model.modules():
        w = getattr(module, "weight", None)
        if w is None or not callable(w):
            continue  # already counted above, or has no weight at all
        wt = w().dequantize() if hasattr(w(), "dequantize") else w()
        total += wt.numel()
        nonzero += (wt != 0).sum().item()
        b_fn = getattr(module, "bias", None)
        if callable(b_fn):
            b = b_fn()
            if b is not None:
                total += b.numel()
                nonzero += (b != 0).sum().item()

    return total, nonzero


def get_model_size_mb(model, path=None):
    """Get model file size in MB. If path given, save and measure file size."""
    if path is None:
        # Estimate from state_dict
        state = model.state_dict()
        size = sum(v.numel() * v.element_size() for v in state.values())
        return size / (1024 * 1024)
    else:
        torch.save(model.state_dict(), path)
        return os.path.getsize(path) / (1024 * 1024)


def evaluate_accuracy(model, dataloader, device):
    """Evaluate top-1 accuracy."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return 100.0 * correct / total


def measure_latency(model, input_size=(1, 3, 32, 32), device='cpu',
                    warmup=20, iterations=100, batch_size=1):
    """
    Measure inference latency with proper warm-up and synchronization.
    Returns mean, std, median latency in ms.
    """
    model.eval()
    model.to(device)
    dummy = torch.randn(batch_size, *input_size[1:]).to(device)

    # Warm-up
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy)
            if device.type == 'cuda':
                torch.cuda.synchronize()

    # Measurement
    latencies = []
    with torch.no_grad():
        for _ in range(iterations):
            if device.type == 'cuda':
                torch.cuda.synchronize()
            start = time.perf_counter()
            _ = model(dummy)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            end = time.perf_counter()
            latencies.append((end - start) * 1000)  # ms

    latencies = np.array(latencies)
    return {
        'mean': float(np.mean(latencies)),
        'std': float(np.std(latencies)),
        'median': float(np.median(latencies)),
        'min': float(np.min(latencies)),
        'max': float(np.max(latencies)),
    }


def measure_peak_memory(model, input_size=(1, 3, 32, 32), device='cpu',
                         batch_size=1, warmup=5, iterations=20):
    """Peak resident memory during repeated inference.

    The original implementation took a single process.memory_info().rss
    sample before one forward pass and one sample after, with no warm-up,
    and returned the (often near-zero or negative) delta as "peak memory" --
    on this pilot's baseline run that produced 0.015625 MB (16 KB) for a
    ~44 MB ResNet-18, which is measurement noise, not a real reading: a
    single before/after RSS delta is dominated by allocator reuse and page
    granularity, not the model's actual footprint.

    This version runs a real warm-up (so one-time allocations like cuDNN
    workspace / first-touch page faults don't land inside the measurement),
    then samples RSS after every iteration and reports the max observed
    value directly (an actual peak, not a single delta).
    """
    model.eval()
    model.to(device)
    dummy = torch.randn(batch_size, *input_size[1:]).to(device)

    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        with torch.no_grad():
            for _ in range(warmup):
                _ = model(dummy)
            torch.cuda.synchronize()
            for _ in range(iterations):
                _ = model(dummy)
            torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB

    import psutil
    process = psutil.Process(os.getpid())
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy)
        peak_rss = process.memory_info().rss
        for _ in range(iterations):
            _ = model(dummy)
            peak_rss = max(peak_rss, process.memory_info().rss)
    return peak_rss / (1024 ** 2)  # MB, absolute peak RSS (not a delta)


def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
