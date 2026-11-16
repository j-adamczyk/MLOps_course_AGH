# Lab 7 - model optimization for inference

This lab concerns optimizing models for inference: making them faster, smaller and cheaper to run in
production. We measure and profile first, and then optimize: compilation, ONNX Runtime, quantization, and,
for language models, the KV cache.

**Learning plan**
1. Measuring inference performance:
   - latency (p50, p95) and throughput, warmup and synchronization
   - evaluation mode, `no_grad()` and `inference_mode()`
   - profiling with `torch.profiler` and Perfetto, padding waste
2. Model compilation with `torch.compile`, recompilations and dynamic shapes
3. ONNX and ONNX Runtime:
   - export with `torch.onnx.export`, dynamic shapes
   - graph optimizations, online and offline
   - execution providers and device selection policies
   - dynamic int8 quantization and its quality
4. Small language models:
   - prefill and decode, TTFT, TPOT, tokens/s
   - KV cache, static cache with `torch.compile`
5. GPU inference (optional): lower precision, CUDA Graphs, pinned memory

**Necessary software**
- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- [Docker](https://docs.docker.com/engine/install/) for the homework

Note that you should also activate `uv` project and install dependencies with `uv sync`.
The main part of the lab runs on CPU. Some exercises also use an NVIDIA GPU (CUDA) or an Apple Silicon GPU
(MPS), and the GPU section is optional.

**Lab**

See [lab instruction](LAB_INSTRUCTION.ipynb), also available as [Markdown](LAB_INSTRUCTION.md).

**Homework**

See [homework instruction](HOMEWORK.md).

**Models**

In the lab, we will be using the [multi-qa-mpnet-base-cos-v1](https://huggingface.co/sentence-transformers/multi-qa-mpnet-base-cos-v1)
sentence embedding model and the [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) language model.

**Additional resources**
- [PyTorch performance tuning guide](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html)
- [A friendly introduction to machine learning compilers and optimizers](https://huyenchip.com/2021/09/07/a-friendly-introduction-to-machine-learning-compilers-and-optimizers.html)
- [ONNX Runtime documentation](https://onnxruntime.ai/docs/)
- [Optimizing LLM inference in transformers](https://huggingface.co/docs/transformers/llm_optims)
