"""Tiny FP16 vs ParoQuant-INT4 latency comparison on MLX (Apple Silicon).

Uses mlx_lm.stream_generate instead of mlx_lm.generate so we get real
per-run TTFT (time to first token) and the library's own generation_tps /
peak_memory instead of only a wall-clock total.

FP16 (Qwen/Qwen3-0.6B) loads via the generic mlx_lm.load(). The ParoQuant
checkpoint (z-lab/Qwen3-0.6B-PARO) stores custom rotation-quantized tensors
(qweight/qzeros/scales/channel_scales/pairs/theta) that mlx_lm.load() does
not understand (raises "parameters not in model"). ParoQuant ships its own
MLX loader for this: paroquant.inference.backends.mlx.load.load(), which
returns an mlx_lm-compatible (model, tokenizer) pair. Two loaders, same
mlx/mlx_lm generation stack underneath, same chip.

Review fix (see phase1.5_review/REVIEW.md): mx.get_peak_memory() is a
process-wide cumulative high-water mark, never reset between model loads.
Running both models sequentially in one process meant PARO_INT4's reported
"peak" was contaminated by FP16's earlier, larger peak -- both benches came
back with the identical value (1.233 GB), which is not a valid per-model
reading. mx.reset_peak_memory() is called at the start of each bench() run,
before that model is even loaded, so each model's peak now reflects only
its own weights-resident-in-memory + generation activations.
"""
import time
import json
import statistics
from pathlib import Path

import mlx.core as mx
from mlx_lm import load as mlx_lm_load
from mlx_lm.generate import stream_generate
from paroquant.inference.backends.mlx.load import load as paro_mlx_load

PROMPT = "2 + 2"
MAX_TOKENS = 32
WARMUP = 3
N = 10

MODELS = {
    "FP16": ("Qwen/Qwen3-0.6B", "mlx_lm"),
    "PARO_INT4": ("z-lab/Qwen3-0.6B-PARO", "paroquant"),
}


def load_model(repo, loader):
    if loader == "mlx_lm":
        return mlx_lm_load(repo)
    model, tokenizer, is_vlm = paro_mlx_load(repo)
    assert not is_vlm, "expected a text-only model"
    return model, tokenizer


def run_one(model, tokenizer, prompt):
    t0 = time.perf_counter()
    ttft = None
    text_parts = []
    last_resp = None
    for resp in stream_generate(model, tokenizer, prompt=prompt, max_tokens=MAX_TOKENS):
        if ttft is None:
            ttft = time.perf_counter() - t0
        text_parts.append(resp.text)
        last_resp = resp
    total = time.perf_counter() - t0
    # NOTE (documented, not fixed -- see REVIEW.md): if MAX_TOKENS were ever
    # 0, or a model emitted EOS with zero generated tokens, last_resp would
    # stay None here and every field below would be None, which would later
    # crash statistics.median() in bench(). Not fixed because MAX_TOKENS is
    # a hardcoded module constant (32) in this tiny benchmark, not user
    # input -- the condition cannot occur as the script is actually invoked.
    return {
        "ttft_s": ttft,
        "total_s": total,
        "generation_tps": last_resp.generation_tps if last_resp else None,
        "prompt_tps": last_resp.prompt_tps if last_resp else None,
        "peak_memory_gb": last_resp.peak_memory if last_resp else None,
        "generation_tokens": last_resp.generation_tokens if last_resp else None,
        "text": "".join(text_parts),
    }


def bench(name, repo, loader):
    mx.reset_peak_memory()  # isolate this model's peak from any prior model's
    model, tokenizer = load_model(repo, loader)
    messages = [{"role": "user", "content": PROMPT}]
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)

    for _ in range(WARMUP):
        run_one(model, tokenizer, prompt)

    runs = [run_one(model, tokenizer, prompt) for _ in range(N)]

    ttfts = [r["ttft_s"] for r in runs]
    tps = [r["generation_tps"] for r in runs]
    totals = [r["total_s"] for r in runs]
    peak_mem = [r["peak_memory_gb"] for r in runs if r["peak_memory_gb"] is not None]

    return {
        "id": name,
        "repo": repo,
        "loader": loader,
        "n": N,
        "max_tokens": MAX_TOKENS,
        "median_ttft_s": statistics.median(ttfts),
        "mean_ttft_s": statistics.mean(ttfts),
        "std_ttft_s": statistics.pstdev(ttfts),
        "median_tok_per_s": statistics.median(tps),
        "mean_tok_per_s": statistics.mean(tps),
        "std_tok_per_s": statistics.pstdev(tps),
        "median_total_s": statistics.median(totals),
        "peak_memory_gb_max": max(peak_mem) if peak_mem else None,
        "sample_output": runs[-1]["text"][:200],
    }


if __name__ == "__main__":
    Path("results").mkdir(exist_ok=True)
    out_path = Path("results/fp16_vs_paro.json")
    results = []
    for name, (repo, loader) in MODELS.items():
        r = bench(name, repo, loader)
        results.append(r)
        out_path.write_text(json.dumps(results, indent=2))
        print(json.dumps(r, indent=2))
