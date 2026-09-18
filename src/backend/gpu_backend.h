#pragma once

#include "../model_config.h"
#include "../gpu_weights.cuh"
#include "../gpu_run_state.h"  
#include "kernel.cuh"

#include <cstddef>
#include <cstdint>
#include <vector>

#include <cublas_v2.h>

class GPUBackend {
public:
    GPUBackend(
        const ModelConfig& config, 
        const GPUModelWeights& weights
    );  
    ~GPUBackend();

    GPUBackend(const GPUBackend&) = delete;
    GPUBackend& operator=(const GPUBackend&) = delete;

    GPUBackend(GPUBackend&&) noexcept;  

    // config_ and weights_ are references, 
    // so they cannot be reassigned.    
    GPUBackend& operator=(GPUBackend&&) noexcept = delete;

    void ResetState() { run_state_.reset(); };  
    const GPURunState& State() const { return run_state_; }

    // Transformer Operations
    void TokenEmbedding(const std::vector<int>& tokens);
    void TokenEmbedding(int token_id);  
    
    void RmsNorm(const DeviceBuffer<__nv_bfloat16>& weight, int seq_len);

    // 後續合併成一個乘法   
    void QkvProjection(int layer, int seq_len);    
    
    void ApplyRoPE(int position, int seq_len);  
  
    void StoreKvCache(int layer, int start_position, int seq_len);  

    void Attention(
        int layer, int start_position, int query_len, int kv_seq_len
    );

    void ResidualAdd(int seq_len);    

    void Mlp(int layer, int seq_len);

    void ComputeLogits(int seq_len);  
    
    void Synchronize() const;
 
    std::vector<float> FetchLogits(int seq_len) const;

    int ArgMaxToken(int seq_len);

private:  
    void AttentionScore(int layer, int query_len, int kv_seq_len);
    void AttentionSoftmax(
        int start_position, int query_len, int kv_seq_len
    );
    void AttentionWeightedSum(int layer, int query_len, int kv_seq_len);
    void AttentionOutput(int layer, int seq_len);
    
    // Matrix operations
    void GemmBf16(
        const __nv_bfloat16* A,
        const __nv_bfloat16* B,
        __nv_bfloat16* C,
        int m,
        int n,  
        int k
    );

    // for compute logits   
    void GemmBf16ToF32(
        const __nv_bfloat16* A,
        const __nv_bfloat16* B,
        float* C,
        int m,  
        int n,
        int k
    );  
    
    void GemmStridedBatchedBf16(
        const __nv_bfloat16* A,
        const __nv_bfloat16* B,
        float* C,  

        int m,
        int n,
        int k,

        long long lda,
        long long ldb,
        long long ldc,

        long long stride_A,
        long long stride_B,
        long long stride_C,

        int batch_count
    );

private:
    const ModelConfig& config_;
    const GPUModelWeights& weights_;
    GPURunState run_state_;

    const int head_dim_;
    const int kv_dim_;
    const int heads_per_kv_;

    cudaStream_t stream_ = nullptr;
    
    cublasHandle_t cublas_handle_ = nullptr;
};