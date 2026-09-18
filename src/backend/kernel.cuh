#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h> 
#include "../device_buffer.cuh"

using bf16 = __nv_bfloat16;  

void LaunchTokenEmbedding(  
    const DeviceBuffer<int>& tokens,
    const DeviceBuffer<bf16>& embed_tokens,
    DeviceBuffer<bf16>& output,    
    int seq_len,
    int hidden_size,
    cudaStream_t stream  
);

void LaunchRmsNorm(
    const bf16* input,
    bf16* output,
    const bf16* weight,
    bf16 eps,
    int seq_len,
    int hidden_size,
    cudaStream_t stream  
);
  
void LaunchApplyRoPE(
    bf16* q,
    bf16* k,
    int seq_len,
    int head_dim,
    int n_heads,
    int n_kv_heads,
    int start_position,
    float rope_theta,
    cudaStream_t stream  
);
  
void LaunchStoreKvCache(
    const bf16* __restrict__ k,
    const bf16* __restrict__ v,
    bf16* __restrict__ k_cache,
    bf16* __restrict__ v_cache,
    int layer,  
    int start_position,
    int seq_len,
    int max_seq_len,
    int n_kv_heads,
    int head_dim,  
    cudaStream_t stream  
);    

void LaunchAttentionSoftmax(
    float* att,
    bf16* output,

    int start_position,
    int query_len,
    int kv_seq_len,

    int head_dim,
    int n_heads,

    cudaStream_t stream  
);

void LaunchResidualAdd(
    bf16* x, 
    const bf16* xb, 
    int seq_len, 
    int hidden_size,
    cudaStream_t stream 
);    
  
void LaunchSwiGlu(
    bf16* gate, 
    const bf16* up, 
    int seq_len, 
    int intermediate_size,
    cudaStream_t stream  
); 

void LaunchArgMaxKernel(
    const float* last_logits,
    int vocab_size,
    int* result,
    cudaStream_t stream
);