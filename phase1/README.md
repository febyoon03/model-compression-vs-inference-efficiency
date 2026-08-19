# Phase 1 — Compression vs. Efficient Execution: A CIFAR-ResNet18 Pilot

Pruning and quantization pilot on CIFAR-10 / ResNet-18, measuring whether parameter/size reduction actually translates into lower inference latency.

## Motivation

It's easy to report "X% of parameters pruned" and stop there. This pilot instead tracks three separate numbers for every compression config — parameter count, on-disk size, and measured latency — because they don't always move together, and the gaps between them are the actual finding.

## Approach

- **Model:** CIFAR-ResNet18, 11.17M parameters, trained from scratch (5-epoch FP32 baseline).
- **Compression configs, all with the same 3-epoch fine-tuning budget:**
  - Unstructured pruning, 40% / 60% (weights zeroed in place, dense tensor shape unchanged)
  - Structured pruning, 40% / 60% (channels physically removed, surviving weights transferred to a narrower network)
  - INT8 post-training dynamic quantization (QNNPACK backend, no fine-tuning)
- **Measured for every config:** parameter count, checkpoint size (MB), CPU/Apple Silicon (arm64) latency, throughput.

## Results

| Config | Params | Size | Latency | Notes |
|---|---|---|---|---|
| Baseline FP32 | 11.17M | — | — | 5-epoch pilot budget |
| Structured 40% | -64% | 15.42MB (-64%) | 4.36ms (-13%) | |
| Structured 60% | -84% | 6.88MB | 2.75ms (-45%, 93.2→197.4 img/s) | |
| Unstructured 40% | nonzero params drop | unchanged (dense shape kept) | 9.32ms (**+86% slower**) | |
| Unstructured 60% | nonzero params drop | unchanged | ~5.14ms (roughly unchanged) | |
| INT8 dynamic quant | ~unchanged (11.17M→11.17M) | ~unchanged (42.70→42.69MB) | median 5.02→5.33ms (**+6% slower**), throughput 93.2→87.3 img/s (**-6%**) | Only `Linear` layers convert on this backend; ResNet-18 is almost all `Conv2d` |

**Takeaway:** structured pruning is the only method here where "smaller" reliably means "faster," because it's the only one that changes the actual tensor shapes the hardware executes. Unstructured pruning made the model *slower* at 40% — zeroed-but-dense weights don't skip any compute on standard CPU kernels, and the sparsity itself adds overhead. INT8 dynamic quantization didn't help either — it left size essentially unchanged and made inference slightly slower, since only the `Linear` layers convert on this backend and ResNet-18 is almost entirely `Conv2d`.

Raw output for every config: [`results/`](results/) (`baseline_results.json`, `structured_40.json`, `structured_60.json`, `unstructured_40.json`, `unstructured_60.json`, `int8_results.json`).

## Tech stack

Python, PyTorch, torchvision, NumPy.

## How to run

```bash
cd phase1
pip install -r requirements.txt
python scripts/run_pilot.py   # runs baseline -> U40 -> S40 -> U60 -> S60 -> INT8, in order
```
`checkpoints/`, `data/`, and `results/` are created automatically (git-ignored). CIFAR-10 downloads automatically via `torchvision` on first run. Each stage can also be run individually, e.g. `python scripts/prune_structured.py --amount 0.6`.

## Limitations / what's next

- 5-epoch baseline and 3-epoch fine-tuning is a pilot-scale budget, not a fully converged training run — absolute accuracy numbers are secondary to the *relative* size/latency comparisons here.
- INT8 dynamic quantization only converts `Linear` layers on this backend; a static or QAT approach that also quantizes `Conv2d` would likely show a very different result and is a natural next step.
- This code went through a review pass that caught and fixed a handful of correctness bugs (a pipeline-failure-silently-reported-as-success bug, a parameter-undercount in the quantized-model measurement, and a couple of dead-code/validation gaps) — the code in `scripts/` here already has those fixes applied.

## Credits

Built independently; no external code reused beyond PyTorch/torchvision itself.
