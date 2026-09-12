#include "kernel.cuh"
#include <cmath> // -INFINITY

namespace {

__global__ 
void TokenEmbedding (
    const int* __restrict__ tokens,
    const bf16* __restrict__ embed_tokens,
    bf16* __restrict__ output,  
    const int seq_len,    
    const int hidden_size) 
{
    // 一個 thread 搬運 8 個 bf16
    int group = blockIdx.x * blockDim.x + threadIdx.x;
    int groups_per_token = hidden_size / 8;
    int total_groups = seq_len * groups_per_token;
    if(group >= total_groups) return;
    
    int token_pos = group / groups_per_token; 
    int group_idx = group % groups_per_token;

    int token_id = tokens[token_pos];  

    const uint4* src = 
        reinterpret_cast<const uint4*>(embed_tokens) + 
        token_id * groups_per_token + 
        group_idx;  
    
    uint4* dst = reinterpret_cast<uint4*>(output) + group;

    *dst = *src;
}

template<int N>
__device__
__forceinline__ float WarpReduceSum(float val) {
    // N 必須是 1~32 的 2 次方
    static_assert(N >= 1 && N <= 32);
    static_assert((N & (N - 1)) == 0);

    if constexpr (N >= 32) 
        val += __shfl_xor_sync(0xffffffff, val, 16);
    
    if constexpr (N >= 16) 
        val += __shfl_xor_sync(0xffffffff, val, 8);
    
    if constexpr (N >= 8) 
        val += __shfl_xor_sync(0xffffffff, val, 4);
    
    if constexpr (N >= 4) 
        val += __shfl_xor_sync(0xffffffff, val, 2);

    if constexpr (N >= 2) 
        val += __shfl_xor_sync(0xffffffff, val, 1);

    return val;
}

__device__ 
__forceinline__ float BlockReduceSum(float val) {
    constexpr int WARP_SIZE = 32;
    constexpr int BLOCK_SIZE = 256;
    constexpr int WARPS_PER_BLOCK = BLOCK_SIZE / WARP_SIZE;

    val = WarpReduceSum<WARP_SIZE>(val);

    __shared__ float reduce_sum[WARPS_PER_BLOCK];

    int lane_id = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;
    
    if(lane_id == 0) {
        reduce_sum[warp_id] = val;
    }

    __syncthreads();

    if(warp_id == 0) {
        val = (lane_id < WARPS_PER_BLOCK) 
            ? reduce_sum[lane_id] 
            : 0.0f;

        val = WarpReduceSum<WARPS_PER_BLOCK>(val);
    }

    return val;  
}

__global__  
void RmsNorm (
    const bf16* __restrict__ input,  
    bf16* __restrict__ output,  
    const bf16* __restrict__ weight,
    bf16 eps,  
    int hidden_size
)  
{
    // one block handle one token
    const int token_idx = blockIdx.x;

    const bf16* x = input + token_idx * hidden_size;
    bf16* y = output + token_idx * hidden_size;

    float sum = 0.0f;

#pragma unroll
    for(int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float value = __bfloat162float(x[i]);
        sum += value * value;
    } 

    sum = BlockReduceSum(sum); // Σx²

    __shared__ float inv_rms;
    if (threadIdx.x == 0) {
        inv_rms = rsqrtf(
            sum / static_cast<float>(hidden_size) + __bfloat162float(eps)
        );
    }
    __syncthreads();

#pragma unroll
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float value = __bfloat162float(x[i]);
        float scale = __bfloat162float(weight[i]);
        float result = value * inv_rms * scale;
        y[i] = __float2bfloat16(result);
    }
}

__global__
void ApplyRoPE (
    bf16* q,
    bf16* k, 
    int seq_len,
    int head_dim,
    int n_heads,
    int n_kv_heads,
    int start_position,
    float rope_theta
) {
    // one thread handle one pair
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    const int half_dim = head_dim / 2;

    // q
    const int q_pairs_per_token = n_heads * half_dim; 
    const int q_total_pairs = seq_len * q_pairs_per_token; 

    if(idx < q_total_pairs) {
        const int token_pos = idx / q_pairs_per_token; 
        const int token_idx = idx % q_pairs_per_token; 

        const int head = token_idx / half_dim;
        const int pair = token_idx % half_dim;

        const int head_offset = 
            token_pos * n_heads * head_dim + head * head_dim;

        // ------------------------------------------------------------
        // RoPE pair:
        // pair 0  → dimension 0  <-> 32
        // pair 1  → dimension 1  <-> 33
        // ...
        // pair 31 → dimension 31 <-> 63
        // ------------------------------------------------------------
        const int index0 = head_offset + pair;
        const int index1 = head_offset + pair + half_dim;

        const float x0 = __bfloat162float(q[index0]);
        const float x1 = __bfloat162float(q[index1]); 

        const float inv_freq =
            1.0f / powf(
                rope_theta,
                2.0f * static_cast<float>(pair)
                / static_cast<float>(head_dim)
            );

        const float angle =
            static_cast<float>(start_position + token_pos) * inv_freq;

        const float s = sinf(angle);
        const float c = cosf(angle);

        // rotation
        const float y0 = x0 * c - x1 * s;
        const float y1 = x1 * c + x0 * s;  

        q[index0] = __float2bfloat16(y0);
        q[index1] = __float2bfloat16(y1); 
    }

    // k
    int k_pairs_per_token = n_kv_heads * half_dim;
    int k_total_pairs = seq_len * k_pairs_per_token;
    if(idx < k_total_pairs) {
        const int token_pos = idx / k_pairs_per_token;  
        const int token_idx = idx % k_pairs_per_token;  

        const int head = token_idx / half_dim;
        const int pair = token_idx % half_dim;

        const int head_offset = 
            token_pos * n_kv_heads * head_dim + head * head_dim;  

        // rope pair
        const int index0 = head_offset + pair;
        const int index1 = head_offset + pair + half_dim;

        const float x0 = __bfloat162float(k[index0]);
        const float x1 = __bfloat162float(k[index1]); 

        const float inv_freq =
            1.0f / powf(
                rope_theta,
                2.0f * static_cast<float>(pair)
                / static_cast<float>(head_dim)
            );

        const float angle =
            static_cast<float>(start_position + token_pos) * inv_freq;

        const float s = sinf(angle);
        const float c = cosf(angle);

        // rotation
        const float y0 = x0 * c - x1 * s;
        const float y1 = x1 * c + x0 * s;  

        k[index0] = __float2bfloat16(y0);
        k[index1] = __float2bfloat16(y1); 
    }
}

__global__
void StoreKvCache(
    const bf16* __restrict__ k,
    const bf16* __restrict__ v,
    bf16* __restrict__ k_cache,
    bf16* __restrict__ v_cache,
    const int layer,  
    const int start_position,
    const int max_seq_len,  
    const int n_kv_heads,
    const int head_dim
) {
    const int token_idx = blockIdx.x;
    
    const int kv_dim = n_kv_heads * head_dim;  
    const int layer_offset = layer * max_seq_len * kv_dim;
    const int position_offset = (start_position + token_idx) * kv_dim;
    const int cache_idx = layer_offset + position_offset + threadIdx.x;
  
    const int src_idx = token_idx * kv_dim + threadIdx.x;

    k_cache[cache_idx] = k[src_idx];
    v_cache[cache_idx] = v[src_idx];       
}

// DS required for Online Softmax
struct __align__(8) MD {
    float m;
    float d;
};

__device__
MD MergeMD(MD a, MD b) {
    // 處理 { -INFINITY, 0.0f } 的情況
    if (a.d == 0.0f) return b;
    if (b.d == 0.0f) return a;  

    const float m = fmaxf(a.m, b.m);
    const float d = a.d * __expf(a.m - m) + b.d * __expf(b.m - m);

    return { m, d };
}

template <const int kWarpSize = 32>
__device__ __forceinline__ MD WarpReduceMD(MD value) {
    constexpr unsigned int mask = 0xffffffff;

#pragma unroll
    for (int stride = kWarpSize >> 1; stride >= 1; stride >>= 1) {
        MD other;
        other.m = __shfl_xor_sync(mask, value.m, stride);
        other.d = __shfl_xor_sync(mask, value.d, stride);

        value = MergeMD(value, other); 
    }  

    return value;
}

__global__
void AttentionSoftmax(
    float* att,
    bf16* output,

    int start_position,
    int query_len,
    int kv_seq_len,

    int head_dim
) {
    // 一個 block 處理：
    //
    // [一個 head][一個 query token]
    //
    // 整條 key dimension

    const int token_idx = blockIdx.x;
    const int head_idx  = blockIdx.y;

    // Absolute position of this query token.
    // Prefill: start_position = 0
    // Decode: start_position = current position
    const int query_position = start_position + token_idx;


    /*
        att layout: [head][query_token][key_token]
        One row: [kv_seq_len]
    */

    float* score =
        att
        + (
            static_cast<std::size_t>(head_idx)
            * query_len
            + token_idx
          )
          * kv_seq_len;


    // scale = 1 / sqrt(head_dim)
    const float head_dim_rsqrt = 
        rsqrtf(static_cast<float>(head_dim));


    // Local Online Softmax state
    MD local = { -INFINITY, 0.0f };


    /*
        First traversal

        Each thread processes part of:

            score[0 ... kv_seq_len-1]
    */

    for (int i = threadIdx.x; i < kv_seq_len; i += blockDim.x) {
        // Causal mask
        if (i > query_position) {
            score[i] = -INFINITY;
            continue;
        }

        // Scale QK^T
        score[i] *= head_dim_rsqrt;

        // Online Softmax update
        const float old_m = local.m;

        local.m = fmaxf(old_m, score[i]);

        local.d =
            local.d * __expf(old_m - local.m) +
            __expf(score[i] - local.m);
    }

    // Reduce MD across warp
    local = WarpReduceMD(local);


    /*
        Warp 0 writes warp results
    */

    __shared__ MD warp_results[8];

    const int lane_id = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;


    if (lane_id == 0) {
        warp_results[warp_id] = local;
    }

    __syncthreads();

    /*
        Final block reduction
    */

    MD block_result = { -INFINITY, 0.0f };


    if (warp_id == 0) {
        if (lane_id < (blockDim.x + 31) / 32) {
            block_result = warp_results[lane_id];
        }

        block_result = WarpReduceMD(block_result);
    }

    // Store global m / d
    __shared__ float row_m;
    __shared__ float row_d;

    if (threadIdx.x == 0) {
        row_m = block_result.m;
        row_d = block_result.d;
    }

    __syncthreads();

    // Second traversal
    // softmax(x_i) = exp(x_i - m) / d
    for (int i = threadIdx.x; i < kv_seq_len; i += blockDim.x) {

        float softmax_value =
            i <= query_position ? 
            __expf(score[i] - row_m) / row_d :
            0.0f;  

        /*
            Keep FP32 result
        */

        score[i] = softmax_value;


        /*
            Store BF16 version
        */

        const std::size_t output_idx =
            (
                static_cast<std::size_t>(head_idx)
                * query_len
                + token_idx
            )
            * kv_seq_len
            + i;

        output[output_idx] = __float2bfloat16(softmax_value);
    }
}

__global__
void ResidualAdd(bf16* x, const bf16* xb, const int hidden_size) {
    const int token_idx = blockIdx.x;

    for(int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        const std::size_t offset =
            static_cast<std::size_t>(token_idx) * hidden_size + i; 

        // 讀取當前數值並轉為 FP32 進行精確計算
        const float val_x  = __bfloat162float(x[offset]);
        const float val_xb = __bfloat162float(xb[offset]);

        // 用 FP32 累加後再寫回 bfloat16（避免半精度累加造成的 0.0625 精度斷層）
        x[offset] = __float2bfloat16(val_x + val_xb);

        //x[offset] += xb[offset];
    }
}  

__global__
void SwiGlu(bf16* gate, const bf16* up, const int intermediate_size) {
    // One block handles one token
    const int token_idx = blockIdx.x;
    const int block_offset = token_idx * intermediate_size;

    for (int i = threadIdx.x; i < intermediate_size; i += blockDim.x) {
        const int offset = block_offset + i;

        // BF16 -> FP32
        const float g = __bfloat162float(gate[offset]);
        const float u = __bfloat162float(up[offset]);

        const float sigmoid = 1.0f / (1.0f + __expf(-g)); // sigmoid(g)
        const float silu = g * sigmoid; // SiLU(g)
        const float result = silu * u; // SiLU(g) * up

        gate[offset] = __float2bfloat16(result); // FP32 -> BF16
    }
}

} // namespace

void LaunchTokenEmbedding(
    const DeviceBuffer<int>& tokens,
    const DeviceBuffer<bf16>& embed_tokens,
    DeviceBuffer<bf16>& output,    
    int seq_len,
    int hidden_size,
    cudaStream_t stream  
)
{
    constexpr int threads_per_block = 256;

    const int groups_per_token = hidden_size / 8;

    const int total_groups = seq_len * groups_per_token;

    const int blocks =
        (total_groups + threads_per_block - 1) / threads_per_block;

    TokenEmbedding<<<blocks, threads_per_block, 0, stream>>>(
        tokens.data(),
        embed_tokens.data(),
        output.data(),
        seq_len,
        hidden_size
    );  

    //CHECK_CUDA(cudaGetLastError());
}

void LaunchRmsNorm(
    const bf16* input,
    bf16* output,
    const bf16* weight,
    const bf16 eps,
    const int seq_len, 
    const int hidden_size,
    cudaStream_t stream  
)
{
    constexpr int threads_per_block = 256;

    RmsNorm<<<seq_len, threads_per_block, 0, stream>>>(
        input,
        output,
        weight,
        eps,
        hidden_size
    );

    //CUDA_CHECK(cudaGetLastError());
}

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
) {
    constexpr int BLOCK_SIZE = 256;

    const int half_dim = head_dim / 2;

    const int q_pairs = seq_len * n_heads * half_dim;

    const int k_pairs = seq_len * n_kv_heads * half_dim;

    const int total_pairs = std::max(q_pairs, k_pairs);

    if (total_pairs == 0) return;

    const int grid_size = 
        (total_pairs + BLOCK_SIZE - 1) / BLOCK_SIZE;

    ApplyRoPE<<<grid_size, BLOCK_SIZE, 0, stream>>>(  
        q,
        k,
        seq_len,
        head_dim,
        n_heads,
        n_kv_heads,
        start_position,
        rope_theta  
    );

    //CHECK_CUDA(cudaGetLastError());
}   

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
) {
    if (start_position < 0 ||
        seq_len < 0 ||
        start_position + seq_len > max_seq_len)
    {
        return;  
    }  

    const dim3 block(n_kv_heads * head_dim); 
    const dim3 grid(seq_len);  
    StoreKvCache<<<grid, block, 0, stream>>>(
        k, v, k_cache, v_cache, 
        layer, start_position, max_seq_len, n_kv_heads, head_dim
    );  
}

void LaunchAttentionSoftmax(
    float* att,
    bf16* output,

    int start_position,
    int query_len,
    int kv_seq_len,

    int head_dim,
    int n_heads,

    cudaStream_t stream  
) {
    constexpr int BLOCK_SIZE = 256;

    const dim3 block(BLOCK_SIZE);
  
    // [query token][head]
    const dim3 grid(query_len, n_heads);

    AttentionSoftmax<<<grid, block, 0, stream>>>(
        att,
        output,

        start_position,
        query_len,
        kv_seq_len,

        head_dim
    );
}

void LaunchResidualAdd(
    bf16* x, 
    const bf16* xb, 
    int seq_len, 
    int hidden_size,
    cudaStream_t stream  
) {
    const dim3 block(256);  
    const dim3 grid(seq_len);
    ResidualAdd<<<grid, block, 0, stream>>>(x, xb, hidden_size);
}  
 
void LaunchSwiGlu(
    bf16* gate, 
    const bf16* up, 
    int seq_len, 
    int intermediate_size,
    cudaStream_t stream  
) {
    const dim3 block(256);
    const dim3 grid(seq_len);

    SwiGlu<<<grid, block, 0, stream>>>(gate, up, intermediate_size);
}