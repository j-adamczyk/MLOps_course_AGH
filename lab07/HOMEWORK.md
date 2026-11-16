# Laboratory 7 homework

Homework takes the models from the lab one step further: the sentence embedding model to a deployment
image, and the small language model to a dedicated local runtime. In both parts, you will measure what you
gain compared to plain PyTorch.

Instructions:
- do tasks in order, they have been designed to build upon prior steps
- write clean code, descriptive variable and function names, use comments where necessary etc.
- send your resulting code as .zip file, with a short report in Markdown with the results (tables) and
  your comments

1. Prepare an embedding service in two versions:
   - a [FastAPI](https://fastapi.tiangolo.com/) application, like in lab 1, with a `POST /embed` endpoint
     that takes a list of texts and returns their embeddings
   - version A: PyTorch with the `SentenceEncoder` from the lab; install the CPU-only PyTorch build
     (`https://download.pytorch.org/whl/cpu` index) to avoid downloading CUDA libraries
   - version B: the best quantized ONNX model from Exercise 4, run with `onnxruntime`, and the
     [tokenizers](https://huggingface.co/docs/tokenizers/) library for tokenization, without PyTorch and
     transformers; enable padding and truncation to 512 tokens in the tokenizer
   - use the same number of threads in both versions, e.g. 4
   - build a Docker image for each version, and copy the model files into the image
2. Compare the two images:
   - image size (`docker images`)
   - cold start: time from `docker run` to the first successful response
   - latency p50 and p95 of 100 requests with 1 text, and of 100 requests with 32 texts, measured with a
     small client script outside the container
   - comment, which version would you deploy, and why? What would change if the service ran on a GPU?
3. Run Qwen3-0.6B with a dedicated local runtime:
   - [llama.cpp](https://github.com/ggml-org/llama.cpp) with the official
     [GGUF model](https://huggingface.co/Qwen/Qwen3-0.6B-GGUF) (8-bit, `Q8_0`); `llama-server` reports
     the prompt and generation timings in its responses
   - on Apple Silicon, you can use [mlx-lm](https://github.com/ml-explore/mlx-lm) with the
     [4-bit](https://huggingface.co/mlx-community/Qwen3-0.6B-4bit) and
     [8-bit](https://huggingface.co/mlx-community/Qwen3-0.6B-8bit) models instead
   - use the same prompt as in the lab, greedy decoding, thinking disabled, and 256 new tokens
4. Compare the runtime with transformers from the lab:
   - model size on disk, TTFT, tokens/s
   - estimate the upper bound of tokens/s from the memory bandwidth of your device and the model size, and
     compare it with the measurements
   - compare the answers, is there a visible quality difference?
   - comment, when would you choose transformers, and when a dedicated runtime?

Grading:
- section 1: 3 points
- section 2: 3 points
- section 3: 2 points
- section 4: 2 points
