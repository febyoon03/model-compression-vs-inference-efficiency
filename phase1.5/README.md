# Phase 1.5 — Reproducing Z-Lab's ParoQuant on Qwen3-0.6B

A direct reproduction of Z-Lab's own open-source ParoQuant method, benchmarking FP16 vs. their INT4 checkpoint on Apple Silicon.

**This is a reproduction of Z-Lab's published method, not a new technique.** See [Credits](#credits).

## Motivation

Phase 1 found that a smaller/sparser model isn't automatically a faster one on CPU. This phase asks the same question one level up the stack, on a real LLM, using a real published quantization method instead of a from-scratch pruning experiment.

## Approach

- Reproduced [z-lab/paroquant](https://github.com/z-lab/paroquant) at commit `f74a96c` (ICLR 2026, pairwise-rotation INT4 quantization).
- Compared `Qwen/Qwen3-0.6B` (FP16, loaded via `mlx_lm`) against `z-lab/Qwen3-0.6B-PARO` (INT4, loaded via ParoQuant's own MLX loader) on the same Apple Silicon machine, same MLX generation stack underneath.
- Measured time-to-first-token (TTFT) and decode throughput over 10 runs each (3 warmup), using `mlx_lm.generate.stream_generate` rather than `mlx_lm.generate` so per-run TTFT is actually captured instead of only a wall-clock total.
- One measurement bug was caught and fixed during this reproduction: `mx.get_peak_memory()` is a process-wide high-water mark that isn't reset between model loads, so running both models sequentially in one process contaminated the second model's "peak memory" reading with the first model's larger peak. Fixed by calling `mx.reset_peak_memory()` immediately before each model loads.

## Results

| | FP16 | ParoQuant INT4 |
|---|---|---|
| Checkpoint size | 1.4GB | 550MB (**-61%**) |
| Median TTFT | 0.216s | 0.219s (~unchanged) |
| Decode throughput | 72.75 tok/s | 49.85 tok/s (**-31%, slower**) |

The checkpoint-size reduction is real and large. The speed result is the opposite of what compression-ratio-on-paper would suggest — INT4 is decoding *slower* than FP16 on this model at this size.

Raw output: [`results/fp16_vs_paro.json`](results/fp16_vs_paro.json). One caveat on that file: its `peak_memory_gb_max` field is identical for both models, which is the exact symptom of the pre-fix `mx.get_peak_memory()` contamination bug described above — this particular run likely predates that fix. TTFT and throughput are unaffected by that bug (it only touches the memory reading), so those numbers are trusted; peak memory is not reported here for that reason.

**Hypothesis** (not independently isolated in this repo): ParoQuant's `RotateQuantizedLinear` performs a rotation + dequantization on every forward pass. At 0.6B parameters, that per-forward overhead plausibly outweighs the memory-bandwidth savings from the smaller weights. Isolating this would need a controlled sweep across model sizes and/or a kernel-level profile of the rotation step in isolation — not done here.

## Tech stack

Python, MLX, mlx-lm, ParoQuant (Z-Lab).

## How to run

```bash
cd phase1.5
pip install -r requirements.txt
# ParoQuant is not on PyPI — clone and install it separately:
git clone https://github.com/z-lab/paroquant.git
git -C paroquant checkout f74a96c
pip install -e ./paroquant

python tiny_fp16_vs_paro.py
```
Requires Apple Silicon (MLX is Apple-Silicon-only). Both models download from Hugging Face on first run. Results are written to `results/fp16_vs_paro.json`.

## Limitations / what's next

- Single model size (0.6B), single prompt, single machine — not a sweep. The open question this leaves is *at what model size or kernel implementation does pairwise-rotation INT4 actually become faster than FP16* — that crossover point is not established here.
- The "why" behind the slowdown is a hypothesis stated above, not something this repo isolates experimentally.

## Credits

This phase reproduces [Z-Lab's ParoQuant](https://github.com/z-lab/paroquant) (commit `f74a96c`) exactly as published — the method, the training, and the `Qwen3-0.6B-PARO` checkpoint are Z-Lab's work, not mine. What's original here is the reproduction itself and the FP16-vs-INT4 TTFT/throughput benchmark built on top of it.
