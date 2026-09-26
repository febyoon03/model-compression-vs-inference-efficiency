# Model Compression vs. Inference Efficiency

Two small studies. Both ask the same question, with two different model families:

**Does a smaller model actually run faster?**

## Motivation

"Smaller model" and "faster model" are not the same thing.

Pruning removes parameters. Quantization uses fewer bits to store each weight. Both can shrink model size. But **parameter count, file size, and inference speed are three different things**.

This project checks whether a smaller model is actually faster, in two settings:

* **Phase 1:** CNN classifier (ResNet-18)
* **Phase 1.5:** Small LLM (Qwen3-0.6B)

## Approach

### Phase 1: ResNet-18 Pruning and Quantization

Trained a ResNet-18 (11.17M parameters) from scratch on CIFAR-10.

Tested five compression methods:

* Unstructured pruning: 40%, 60%
* Structured pruning: 40%, 60%
* INT8 post-training dynamic quantization

Each pruned model got the same fine-tuning budget: 3 epochs.

For each model, measured:

* Parameter count
* Model size on disk
* CPU / Apple Silicon latency
* Accuracy

Structured pruning removes whole channels. Unstructured pruning just zeroes out individual weights, one at a time.

### Phase 1.5: Reproducing ParoQuant

This phase reproduces Z-Lab's open-source method: pairwise-rotation INT4 quantization.

Compared:

* Qwen3-0.6B FP16
* Qwen3-0.6B-PARO INT4

Ran the benchmark on Apple Silicon, using the MLX backend.

This is a reproduction of an existing method, not a new one. See [Credits](#credits).

Full results:

* [`phase1/README.md`](phase1/README.md)
* [`phase1.5/README.md`](phase1.5/README.md)

Raw pilot results for Phase 1: [`phase1/results/`](phase1/results/)

## Results

| Method | Size / Parameter Change | Inference Change |
|---|---:|---:|
| Structured pruning 40% | Size -64% | Latency -13% |
| Structured pruning 60% | Size -84% (6.88MB) | Latency -45% (93 → 197 img/s) |
| Unstructured pruning 40% / 60% | Fewer nonzero parameters, but same tensor shape | U40: 86% slower, U60: about unchanged |
| INT8 dynamic quantization | 42.70 → 42.69MB | 6% slower (93.2 → 87.3 img/s) |
| ParoQuant INT4 (Qwen3-0.6B) | 1.4GB → 550MB (-61%) | Decode 31% slower (72.75 → 49.85 tok/s) |

**Compression ratio alone does not predict speed.**

In Phase 1, unstructured pruning cut the number of nonzero parameters, but the model was not faster. It's still dense under the hood. Structured pruning cut both size and latency, since it actually removes channels.

In Phase 1.5, INT4 shrank the checkpoint by 61%. But decode speed was 31% slower than FP16.

What matters most is how the compressed model actually runs: the tensor shape, and what the hardware can do with it.

## Tech Stack

**Phase 1:** Python, PyTorch, torchvision

**Phase 1.5:** Python, MLX, mlx-lm, ParoQuant (Z-Lab)

## How to Run

See each phase's README for exact commands and setup.

The two phases use different frameworks. They don't share a virtual environment.

## Limitations

* Why INT4 is slower here is still just a hypothesis. Rotation and dequantization overhead might outweigh the bandwidth savings at this size (0.6B parameters). This wasn't tested directly in its own experiment.
* Open question: at what model size, or with what kernel, does pairwise-rotation INT4 become faster than FP16?

## Credits

Phase 1.5 reproduces Z-Lab's ParoQuant method, using commit `f74a96c` of [`z-lab/paroquant`](https://github.com/z-lab/paroquant).

The ParoQuant method and the `Qwen3-0.6B-PARO` checkpoint were made by Z-Lab. This project adds the reproduction and the throughput/TTFT benchmark, not the compression method itself.
