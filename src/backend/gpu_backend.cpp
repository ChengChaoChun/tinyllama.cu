#include "gpu_backend.h"
//#include "kernel.cuh"

#include <vector>
#include <utility>  
#include <stdexcept> 

GPUBackend::GPUBackend(
    const ModelConfig& config, const GPUModelWeights& weights
) : config_(config), 
    weights_(weights), 
    run_state_(
          config.hidden_size,
          config.n_heads,
          config.n_kv_heads,
          config.n_layers,
          config.max_seq_len,  
          config.vocab_size,
          config.intermediate_size
    ),  
    head_dim_(config.hidden_size / config.n_heads),
    kv_dim_(config.n_kv_heads * head_dim_),
    heads_per_kv_(config.n_heads / config.n_kv_heads)
{
    // Create CUDA stream
    cudaError_t cuda_status = cudaStreamCreate(&stream_);
    if (cuda_status != cudaSuccess) {
        throw std::runtime_error(
            "Failed to create CUDA stream: " +
            std::string(cudaGetErrorString(cuda_status))
        );
    }

    // Create cuBLAS handle
    cublasStatus_t cublas_status = cublasCreate(&cublas_handle_);
    if (cublas_status != CUBLAS_STATUS_SUCCESS) {
        cudaStreamDestroy(stream_);
        stream_ = nullptr;

        throw std::runtime_error(
            "Failed to create cuBLAS handle"
        );
    } 

    // Associate cuBLAS with CUDA stream
    cublas_status = cublasSetStream(cublas_handle_, stream_);
    if (cublas_status != CUBLAS_STATUS_SUCCESS) {
        cublasDestroy(cublas_handle_);
        cublas_handle_ = nullptr;

        cudaStreamDestroy(stream_);
        stream_ = nullptr;

        throw std::runtime_error(
            "Failed to set CUDA stream for cuBLAS"
        );
    }
}

GPUBackend::~GPUBackend() {
    // Destroy cuBLAS handle
    if (cublas_handle_ != nullptr) {
        cublasDestroy(cublas_handle_);
        cublas_handle_ = nullptr;
    }  

    // Destroy CUDA stream
    if (stream_ != nullptr) {
        cudaStreamDestroy(stream_);
        stream_ = nullptr;
    }
}

GPUBackend::GPUBackend(GPUBackend&& other) noexcept
    : config_(other.config_),
      weights_(other.weights_),
      run_state_(std::move(other.run_state_)),
      head_dim_(other.head_dim_),
      kv_dim_(other.kv_dim_),
      heads_per_kv_(other.heads_per_kv_),
      stream_(std::exchange(other.stream_, nullptr)),  
      cublas_handle_(std::exchange(other.cublas_handle_, nullptr))
{}      

void GPUBackend::TokenEmbedding(const std::vector<int>& tokens) {
    if (tokens.empty()) return;

    const int seq_len = static_cast<int>(tokens.size());

    run_state_.tokens.copy_from_host(tokens.data(), tokens.size());

    LaunchTokenEmbedding(
        run_state_.tokens, weights_.embed_tokens, run_state_.x,
        seq_len, config_.hidden_size,
        stream_
    );
}  
 
void GPUBackend::TokenEmbedding(int token_id) {
    run_state_.tokens.copy_from_host(&token_id, 1);

    LaunchTokenEmbedding(
        run_state_.tokens, weights_.embed_tokens, run_state_.x,
        1, config_.hidden_size,
        stream_
    );
}  
  
void GPUBackend::RmsNorm( 
    const DeviceBuffer<__nv_bfloat16>& weight, int seq_len
) {
    LaunchRmsNorm(
        run_state_.x.data(), run_state_.xb.data(), weight.data(),
        config_.rms_norm_eps, seq_len, config_.hidden_size,
        stream_
    );  
}
  
void GPUBackend::QkvProjection(int layer, int seq_len) {
    const auto& attention = weights_.layers[layer].self_attn;
  
    // Q
    GemmBf16(
        run_state_.xb.data(), 
        attention.q_proj.data(), 
        run_state_.q.data(),
        seq_len, config_.hidden_size, config_.hidden_size
    );

    // K
    GemmBf16(
        run_state_.xb.data(),
        attention.k_proj.data(),
        run_state_.k.data(),
        seq_len, kv_dim_, config_.hidden_size
    );

    // V
    GemmBf16(  
        run_state_.xb.data(),
        attention.v_proj.data(),
        run_state_.v.data(),
        seq_len, kv_dim_, config_.hidden_size
    );
}

void GPUBackend::ApplyRoPE(int position, int seq_len) {
    LaunchApplyRoPE(
        run_state_.q.data(), run_state_.k.data(),
        seq_len, 
        head_dim_, 
        config_.n_heads, 
        config_.n_kv_heads,
        position,
        config_.rope_theta,
        stream_
    );  
}
  
void GPUBackend::StoreKvCache(
    int layer, int start_position, int seq_len
) {  
    LaunchStoreKvCache(
        run_state_.k.data(), run_state_.v.data(),
        run_state_.k_cache.data(), run_state_.v_cache.data(),
        layer,  
        start_position,  
        seq_len,  
        config_.max_seq_len,
        config_.n_kv_heads,
        head_dim_,
        stream_
    );    
}

void GPUBackend::Attention(
    int layer, int start_position, int query_len, int kv_seq_len
) {
    AttentionScore(layer, query_len, kv_seq_len);
    AttentionSoftmax(start_position, query_len, kv_seq_len);
    AttentionWeightedSum(layer, query_len, kv_seq_len);
    AttentionOutput(layer, query_len);
}  
  
void GPUBackend::AttentionScore(
    int layer, int query_len, int kv_seq_len
) {
    // Q:   [query_len][num_heads][head_dim]
    // K:   [kv_seq_len][num_kv_heads][head_dim]
    // Att: [num_heads][query_len][kv_seq_len]
    const long long q_lda = config_.hidden_size;
    const long long k_ldb = static_cast<long long>(kv_dim_);  
    const long long att_ldc = kv_seq_len;

    const long long q_stride = head_dim_;
    const long long k_stride = 0;
    const long long att_stride =
        static_cast<long long>(query_len) * kv_seq_len; 

    // GQA: each KV head is shared by heads_per_kv_ query heads.
    // Process all query heads associated with a KV head as one
    // strided-batched GEMM operation.
    for (int kv_head = 0; kv_head < config_.n_kv_heads; ++kv_head) {
        const int q_head_begin = kv_head * heads_per_kv_;
        
        const __nv_bfloat16* q_ptr =
            run_state_.q.data() +
            static_cast<std::size_t>(q_head_begin) * head_dim_;
 
        const std::size_t layer_offset =   
            static_cast<std::size_t>(layer) * 
            config_.max_seq_len * kv_dim_;
        const __nv_bfloat16* k_ptr =
            run_state_.k_cache.data() + 
            layer_offset +  
            static_cast<std::size_t>(kv_head) * head_dim_;

        float* att_ptr =  
            run_state_.att.data() +
            static_cast<std::size_t>(q_head_begin) *
            query_len * kv_seq_len; 

        GemmStridedBatchedBf16(
            q_ptr, k_ptr, att_ptr,
            query_len, kv_seq_len, head_dim_, 
            q_lda, k_ldb, att_ldc,    
            q_stride, k_stride, att_stride,   
            heads_per_kv_
        );  
    }
}  

void GPUBackend::AttentionSoftmax(
    int start_position, int query_len, int kv_seq_len
) {
    LaunchAttentionSoftmax(
        run_state_.att.data(), run_state_.att_bf16.data(),
        start_position, query_len, kv_seq_len,
        head_dim_, config_.n_heads,
        stream_
    );
}      

void GPUBackend::AttentionWeightedSum(
    int layer, int query_len, int kv_seq_len
) {
    const float alpha = 1.0f;
    const float beta  = 0.0f;

    // V cache: [layer][position][kv_head][head_dim]
    // Att:     [q_head][query][key]
    // Output:  [query][q_head][head_dim]  
    const long long v_lda = kv_dim_;
    const long long att_ldb = kv_seq_len;
    const long long output_ldc = config_.hidden_size;
    
    const long long v_stride = 0;
    const long long att_stride = 
        static_cast<long long>(query_len) * kv_seq_len;
    const long long output_stride = head_dim_;

    const std::size_t layer_offset =
        static_cast<std::size_t>(layer) * config_.max_seq_len * kv_dim_;

    for (int kv_head = 0; kv_head < config_.n_kv_heads; ++kv_head) {
        const int q_head_begin = kv_head * heads_per_kv_;

        const __nv_bfloat16* V =
            run_state_.v_cache.data()
            + layer_offset
            + static_cast<std::size_t>(kv_head)
              * head_dim_;

        const __nv_bfloat16* Attention =
            run_state_.att_bf16.data()
            + static_cast<std::size_t>(q_head_begin)
              * query_len
              * kv_seq_len;

        __nv_bfloat16* Output =
            run_state_.xb2.data()
            + static_cast<std::size_t>(q_head_begin)
              * head_dim_;

        cublasStatus_t status =
            cublasGemmStridedBatchedEx(
                cublas_handle_,
                CUBLAS_OP_N, CUBLAS_OP_N,     
                head_dim_, query_len, kv_seq_len,   
                &alpha,
                V, CUDA_R_16BF, v_lda, v_stride,
                Attention, CUDA_R_16BF, att_ldb, att_stride,
                &beta,
                Output, CUDA_R_16BF, output_ldc, output_stride,
                heads_per_kv_,
                CUDA_R_32F,
                CUBLAS_GEMM_DEFAULT
            );  

        if (status != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error(
                "attention_weighted_sum: "
                "cublasGemmStridedBatchedEx failed"
            );
        }
    }
}

void GPUBackend::AttentionOutput(const int layer, const int seq_len) {
    GemmBf16(    
        run_state_.xb2.data(),              
        weights_.layers[layer].self_attn.o_proj.data(),
        run_state_.xb.data(),               
        seq_len, config_.hidden_size, config_.hidden_size               
    );
}

void GPUBackend::ResidualAdd(int seq_len) {
    LaunchResidualAdd(  
        run_state_.x.data(), run_state_.xb.data(),
        seq_len, config_.hidden_size,
        stream_
    );
}

void GPUBackend::Mlp(const int layer, const int seq_len) {
    // gate proj
    GemmBf16(
        run_state_.xb.data(),
        weights_.layers[layer].mlp.gate_proj.data(),
        run_state_.hb.data(),
        seq_len, config_.intermediate_size, config_.hidden_size
    );

    //up proj
    GemmBf16(
        run_state_.xb.data(),
        weights_.layers[layer].mlp.up_proj.data(),
        run_state_.hb2.data(),
        seq_len, config_.intermediate_size, config_.hidden_size
    );
  
    LaunchSwiGlu(
        run_state_.hb.data(),
        run_state_.hb2.data(),
        seq_len, config_.intermediate_size,
        stream_  
    );

    GemmBf16(
        run_state_.hb.data(),  
        weights_.layers[layer].mlp.down_proj.data(),
        run_state_.xb.data(),
        seq_len, config_.hidden_size, config_.intermediate_size
    );
}
  
void GPUBackend::ComputeLogits(const int seq_len) {
    GemmBf16ToF32(
        run_state_.xb.data(),
        weights_.lm_head.data(),
        run_state_.logits.data(),
        seq_len, config_.vocab_size, config_.hidden_size  
    );
}

// ============================================================
// GEMM layout and cuBLAS mapping
// ============================================================
// Input matrices are row-major: 
// A: [m, k], B: [n, k], C: [m, n]
// C = A * B^T
//
// cuBLAS uses column-major, so:
// C^T = B * A^T
//
// B is row-major [n, k], so cuBLAS sees B^T [k, n].
// Therefore B needs CUBLAS_OP_T.
//
// A is row-major [m, k], so cuBLAS sees A^T [k, m].
// Therefore A uses CUBLAS_OP_N.
// ============================================================
void GPUBackend::GemmBf16(
    const __nv_bfloat16* A,
    const __nv_bfloat16* B,
    __nv_bfloat16* C,
    int m, int n, int k
)
{
    constexpr float alpha = 1.0f;
    constexpr float beta  = 0.0f;

    const cublasStatus_t status = cublasGemmEx(
        cublas_handle_,
        CUBLAS_OP_T, CUBLAS_OP_N,  // Op B, Op A
        n, m, k,                   
        &alpha,  
        B, CUDA_R_16BF, k,         
        A, CUDA_R_16BF, k,         
        &beta,
        C, CUDA_R_16BF, n,         
        CUDA_R_32F,                
        CUBLAS_GEMM_DEFAULT        
    ); 

    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("cublasGemmEx failed");
    }
}

void GPUBackend::GemmBf16ToF32(
    const __nv_bfloat16* A,
    const __nv_bfloat16* B,
    float* C,
    int m, int n, int k
)
{
    constexpr float alpha = 1.0f;
    constexpr float beta  = 0.0f;

    const cublasStatus_t status = cublasGemmEx(
        cublas_handle_,
        CUBLAS_OP_T, CUBLAS_OP_N,  // Op B, Op A
        n, m, k,      
        &alpha,
        B, CUDA_R_16BF, k,      
        A, CUDA_R_16BF, k,     
        &beta,
        C, CUDA_R_32F, n,      
        CUDA_R_32F,
        CUBLAS_GEMM_DEFAULT
    );

    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(
            "cublasGemmEx BF16->FP32 failed"
        );
    }
}

void GPUBackend::GemmStridedBatchedBf16(
    const __nv_bfloat16* A,
    const __nv_bfloat16* B,
    float* C,  
    int m, int n, int k,
    long long lda, long long ldb, long long ldc,
    long long stride_A, long long stride_B, long long stride_C,    
    int batch_count  
)
{
    constexpr float alpha = 1.0f;
    constexpr float beta  = 0.0f;

    const cublasStatus_t status =
        cublasGemmStridedBatchedEx(
            cublas_handle_,
            CUBLAS_OP_T, CUBLAS_OP_N,  // Op B, Op A
            n, m, k,
            &alpha,
            B, CUDA_R_16BF, ldb, stride_B,
            A, CUDA_R_16BF, lda, stride_A,
            &beta,
            C, CUDA_R_32F, ldc, stride_C,
            batch_count,
            CUDA_R_32F,
            CUBLAS_GEMM_DEFAULT
        );

    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(
            "cublasGemmStridedBatchedEx failed"
        );
    }
}

void GPUBackend::Synchronize() const {
    CHECK_CUDA(cudaDeviceSynchronize());
}

std::vector<float> GPUBackend::FetchLogits(int seq_len) const {
    const int vocab_size = config_.vocab_size;

    const std::size_t count =
        static_cast<std::size_t>(seq_len) *
        static_cast<std::size_t>(vocab_size);

    std::vector<float> host(count);

    cudaError_t status = cudaMemcpy(
        host.data(),
        run_state_.logits.data(),
        count * sizeof(float),
        cudaMemcpyDeviceToHost
    );

    if (status != cudaSuccess) {
        throw std::runtime_error(
            std::string("logits_host cudaMemcpy failed: ")
            + cudaGetErrorString(status)
        );
    }

    return host;
}