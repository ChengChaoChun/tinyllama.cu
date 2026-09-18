#include "device_buffer.cuh" 

// 初始化內存分配可能要改? 也許不用一次分配 max_seq_len (max_seq_len)?
class GPURunState {
public:
    using bf16 = __nv_bfloat16;

    GPURunState(
        int hidden_size,
        int num_heads,
        int num_kv_heads,
        int num_layers,
        int max_seq_len,
        int vocab_size,
        int intermediate_size    
    ) : tokens(max_seq_len),  

        x(max_seq_len * hidden_size),
        xb(max_seq_len * hidden_size),
        xb2(max_seq_len * hidden_size), 

        q(max_seq_len * hidden_size),
        k(max_seq_len * (num_kv_heads * (hidden_size / num_heads))),
        v(max_seq_len * (num_kv_heads * (hidden_size / num_heads))),

        att(static_cast<size_t>(num_heads) * max_seq_len * max_seq_len),
        att_bf16(static_cast<size_t>(num_heads) * max_seq_len * max_seq_len),

        hb(max_seq_len * intermediate_size),
        hb2(max_seq_len * intermediate_size),

        logits(max_seq_len * vocab_size),

        k_cache(static_cast<size_t>(num_layers) * max_seq_len *
                (num_kv_heads * (hidden_size / num_heads))),
        v_cache(static_cast<size_t>(num_layers) * max_seq_len *
                    (num_kv_heads * (hidden_size / num_heads))),
        
        argmax_token_(1)
    {}    

    GPURunState(const GPURunState&) = delete;
    GPURunState& operator=(const GPURunState&) = delete;
    GPURunState(GPURunState&&) noexcept = default;
    GPURunState& operator=(GPURunState&&) noexcept = default;

    void reset() {
        x.memset(0);    
        xb.memset(0);
        xb2.memset(0);

        q.memset(0);
        k.memset(0);
        v.memset(0);

        att.memset(0);
        att_bf16.memset(0);

        hb.memset(0);
        hb2.memset(0);

        logits.memset(0);

        k_cache.memset(0);
        v_cache.memset(0);
    }

public:
    DeviceBuffer<int> tokens;
    DeviceBuffer<bf16> x, xb, xb2;
    DeviceBuffer<bf16> q, k, v;
    DeviceBuffer<float> att;
    DeviceBuffer<bf16> att_bf16; // attention softmax ouput 暫定
    DeviceBuffer<bf16> hb, hb2; // mlp [seq_len * intermediate_size]
    DeviceBuffer<float> logits;
    DeviceBuffer<bf16> k_cache, v_cache;
    DeviceBuffer<int> argmax_token_;
};