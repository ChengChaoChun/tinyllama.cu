#pragma once
#include <vector>
#include <cuda_bf16.h>
#include "device_buffer.cuh"

struct GPUAttentionWeights {
    DeviceBuffer<__nv_bfloat16> q_proj;
    DeviceBuffer<__nv_bfloat16> k_proj;
    DeviceBuffer<__nv_bfloat16> v_proj;
    DeviceBuffer<__nv_bfloat16> o_proj;
};

struct GPUMLPWeights {
    DeviceBuffer<__nv_bfloat16> gate_proj;
    DeviceBuffer<__nv_bfloat16> up_proj;
    DeviceBuffer<__nv_bfloat16> down_proj;
};

struct GPULayerWeights {
    DeviceBuffer<__nv_bfloat16> input_layernorm;
    GPUAttentionWeights self_attn;
    DeviceBuffer<__nv_bfloat16> post_attention_layernorm;
    GPUMLPWeights mlp;
};

struct GPUModelWeights {
    DeviceBuffer<__nv_bfloat16> embed_tokens;
    std::vector<GPULayerWeights> layers;
    DeviceBuffer<__nv_bfloat16> norm;
    DeviceBuffer<__nv_bfloat16> lm_head;
};