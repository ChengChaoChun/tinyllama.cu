# tinyllama.cu
  
<p align="center">
  <img src="assets/logo.png" width="400">
  <br>
  <h3 align="center">A Lightweight C++/CUDA Inference Engine for TinyLlama-1.1B Built from Scratch</h3>
</p>  

## Overview
`tinyllama.cu` implements the core Transformer inference pipeline for TinyLlama-1.1B-Chat-v1.0, with both CPU and NVIDIA GPU backends.
  
Both backends support prompt prefill, autoregressive decoding, and KV caching. The GPU backend additionally uses custom CUDA kernels for specialized operations and cuBLAS for matrix multiplication.  

## Key Features
* **Memory-Mapped Model Loading:** Uses `mmap` to map SafeTensors model weights directly into memory and provides zero-copy tensor views on the host side.

* **GQA & KV Cache:** Implements Grouped-Query Attention (GQA) with dynamic KV caching to accelerate both prompt prefill and autoregressive decoding.

* **Custom CUDA Kernels:** Implements specialized CUDA kernels for key Transformer operations, optimized for GPU inference.

* **cuBLAS Acceleration:** Leverages cuBLAS for high-performance matrix multiplications across linear projections and multi-head attention loops.

## Getting Started
### Requirements
* **OS:** Linux (required for POSIX `mmap()` support)
* **C++ Compiler:** GCC (>= 12.0 recommended, with C++20 support)
* **Build System:** CMake (>= 3.28)
* **CUDA Toolkit:** CUDA Toolkit (>= 12.0), including nvcc
* **GPU Hardware:** NVIDIA GPU with CUDA support and sufficient VRAM to run TinyLlama-1.1B-Chat-v1.0
  * Tested on **RTX 4080 SUPER (16 GB)**
  * Tested on **RTX 2080 Ti (11 GB)**

### Model Setup
Download the **TinyLlama-1.1B-Chat-v1.0** model and place it under the `models/` directory.

The expected directory structure is:

```text
tinyllama.cu/
├── models/
│   └── TinyLlama-1.1B-Chat-v1.0/
│       ├── config.json
│       ├── model.safetensors
│       └── tokenizer.model
├── src/
├── CMakeLists.txt
└── ...
```

The model can be downloaded from Hugging Face using the Hugging Face CLI:

```bash
# Install Hugging Face CLI (if not installed)
pip install -U huggingface_hub

hf download \
    TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --local-dir models/TinyLlama-1.1B-Chat-v1.0
```

### Build
Configure and build the project with CMake:

```bash
cmake -B build
cmake --build build -j
```

### Run
Start the inference engine after building:

```bash
./build/tinyllama-cu
```

The inference engine provides an interactive command-line interface. Enter a prompt when prompted to start text generation.

## Validation
The inference results are verified against PyTorch for numerical correctness and autoregressive generation.

### Logit Verification
* Prompt prefill + 100 decoding steps
* Argmax match: **100/100 (100%)**
* Cosine similarity: **> 0.9995**

### Generation Verification
* Generated tokens: **100**
* Token match: **100/100 (100%)**
* Result: **PASS**

[View the full verification log](test/verification.log)  

## License
This project is licensed under the MIT License.