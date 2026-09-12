#pragma once

#include <vector>
#include <algorithm> 

class RunState {
public:
    RunState(
        int hidden_size,
        int num_heads,
        int num_kv_heads,
        int num_layers,
        int max_seq_len,
        int vocab_size,
        int intermediate_size
    ) {
        int head_dim = hidden_size / num_heads;
        int kv_dim = num_kv_heads * head_dim;

        // Hidden states
        x.resize(hidden_size);
        xb.resize(hidden_size);
        xb2.resize(hidden_size);

        // Attention Q, K, V
        q.resize(hidden_size);
        k.resize(kv_dim);
        v.resize(kv_dim);

        // Attention scores
        att.resize(static_cast<size_t>(num_heads) * max_seq_len);

        // MLP intermediate
        hb.resize(intermediate_size);
        hb2.resize(intermediate_size);

        // Logits
        logits.resize(vocab_size);

        // KV Cache  
        size_t kv_cache_size = 
            static_cast<size_t>(num_layers) * max_seq_len * kv_dim;
        key_cache.resize(kv_cache_size);
        value_cache.resize(kv_cache_size);
    }

    // 預設可轉移 (Moveable) 與複製 (Copyable)
    RunState(const RunState&) = default;
    RunState& operator=(const RunState&) = default;
    RunState(RunState&&) noexcept = default;
    RunState& operator=(RunState&&) noexcept = default;

    void reset() {  
        std::ranges::fill(x, 0.0f);
        std::ranges::fill(xb, 0.0f);
        std::ranges::fill(xb2, 0.0f);

        std::ranges::fill(q, 0.0f);
        std::ranges::fill(k, 0.0f);
        std::ranges::fill(v, 0.0f);

        std::ranges::fill(att, 0.0f);

        std::ranges::fill(hb, 0.0f);
        std::ranges::fill(hb2, 0.0f);

        std::ranges::fill(logits, 0.0f);

        std::ranges::fill(key_cache, 0.0f);
        std::ranges::fill(value_cache, 0.0f);
    }

public:
    std::vector<float> x;
    std::vector<float> xb;
    std::vector<float> xb2;

    std::vector<float> q;
    std::vector<float> k;
    std::vector<float> v;
    std::vector<float> att;

    std::vector<float> hb;
    std::vector<float> hb2;

    std::vector<float> logits;

    std::vector<float> key_cache;
    std::vector<float> value_cache;
};