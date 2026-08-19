# Model Compression vs. Inference Efficiency

Two small studies asking the same question on two different model families: does making a model smaller actually make it faster to run?

## Motivation

"Smaller model" and "faster model" get used almost interchangeably, but they aren't the same claim. Pruning removes parameters, quantization reduces bit-width, and both usually shrink a checkpoint on disk — but disk size, parameter count, and wall-clock latency are three different numbers that don't have to move together. This repo tests that gap directly on two very different setups: a CNN classifier (Phase 1) and a small LLM (Phase 1.5).

## Approach

**Phase 1 — CIFAR-ResNet18 pruning and quantization.** A ResNet-18 (11.17M params) trained from scratch on CIFAR-10, then compressed five ways: unstructured pruning (40%/60%), structured pruning (40%/60%, with real channel removal and weight transfer), and INT8 post-training dynamic quantization. All pruning configs get the same 3-epoch fine-tuning budget. Every variant is measured for parameter count, on-disk size, and CPU/Apple Silicon (arm64) latency — not just accuracy.

**Phase 1.5 — Reproducing Z-Lab's ParoQuant.** A direct reproduction of Z-Lab's own open-source pairwise-rotation INT4 quantization method ([z-lab/paroquant](https://github.com/z-lab/paroquant), commit `f74a96c`), comparing Qwen3-0.6B FP16 against the official Qwen3-0.6B-PARO INT4 checkpoint on Apple Silicon with the MLX backend. This is a reproduction of their published method, not a new technique — see [Credits](#credits).

Full write-up for both phases: [`phase1/README.md`](phase1/README.md), [`phase1.5/README.md`](phase1.5/README.md). Raw pilot output for Phase 1 is committed under [`phase1/results/`](phase1/results/) rather than only summarized here.

## Results

| | Params/size change | Latency / throughput change |
|---|---|---|
| Structured pruning 40% | size -64% | latency -13% |
| Structured pruning 60% | size -64% (6.88MB) | latency -45% (93→197 img/s) |
| Unstructured pruning 40%/60% | nonzero params drop, but dense tensor shape unchanged → size ~unchanged | U40 is 86% **slower**; U60 roughly unchanged |
| INT8 dynamic quantization | ~unchanged (42.70→42.69MB) | **6% slower** (93.2→87.3 img/s; only `Linear` layers convert, ResNet-18 is mostly `Conv2d`) |
| ParoQuant INT4 (Qwen3-0.6B) | checkpoint -61% (1.4GB→550MB) | decode throughput **31% slower** (72.75→49.85 tok/s); TTFT ~unchanged |

Sparsity didn't predict speed in Phase 1, and bit-width didn't predict speed in Phase 1.5 — in both cases, the actual execution path/kernel support decided latency, not the compression ratio on paper. The unstructured-pruning and ParoQuant-INT4 numbers are the two most important rows in this table precisely because they're negative results.

## Tech stack

Phase 1: Python, PyTorch, torchvision. Phase 1.5: Python, MLX, mlx-lm, ParoQuant (Z-Lab).

## How to run

See each phase's own README for exact commands and environment setup — the two phases use unrelated stacks (PyTorch vs. MLX) and don't share a virtual environment.

## Limitations / what's next

- Phase 1.5's "why INT4 is slower at this scale" is a hypothesis (per-forward rotation+dequant overhead outweighing bandwidth savings at 0.6B), not something independently isolated in this repo — see that phase's README for the reasoning and what an isolating experiment would need.
- Open question carried forward: at what model size / kernel implementation does pairwise-rotation INT4 actually cross over to being faster than FP16?

## Credits

Phase 1.5 reproduces Z-Lab's published ParoQuant method exactly (commit `f74a96c` of [z-lab/paroquant](https://github.com/z-lab/paroquant)) — the method and the `Qwen3-0.6B-PARO` checkpoint are theirs, not mine. This repo's contribution in that phase is the reproduction and the throughput/TTFT benchmark, not the compression technique itself.
