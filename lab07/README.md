## Lab 7 - model optimization for inference

This lab focuses on optimizing machine learning models for inference. After training,
models need to be optimized to reduce latency, memory usage, and computational costs
in production environments. We will explore various PyTorch optimization techniques
and conversion to ONNX format for efficient deployment.

Proper optimization can reduce inference time by 2-10 times, significantly lowering
infrastructure costs and improving user experience in production ML systems.

**Learning Plan**
1. PyTorch inference best practices (eval mode, no_grad).
2. torch.compile() for automatic optimization.
3. Model quantization
4. GPU optimization strategies & CUDA 
5. ONNX format for cross-platform deployment

**Necessary software**
- [Docker and Docker Compose](https://docs.docker.com/engine/install/), 
  also [see those post-installation notes](https://docs.docker.com/engine/install/linux-postinstall/)
- [uv](https://docs.astral.sh/uv/getting-started/installation/) for dependency management

**Lab**

See [lab instruction](LAB_INSTRUCTION.md).

There is no homework, only lab this time :)

**Additional Resources**
- [PyTorch Performance Tuning Guide](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html)
- [Hugging Face Optimization Rush](https://huggingface.co/blog/Isayoften/optimization-rush)
- [ML Compilers and Optimizers Introduction](https://huyenchip.com/2021/09/07/a-friendly-introduction-to-machine-learning-compilers-and-optimizers.html)
- [ONNX Docs](https://onnxruntime.ai/docs/)
- [ONNX Graph Optimizations](https://onnxruntime.ai/docs/performance/model-optimizations/graph-optimizations.html)
- [ONNX Execution Providers](https://iot-robotics.github.io/ONNXRuntime/docs/execution-providers/)
