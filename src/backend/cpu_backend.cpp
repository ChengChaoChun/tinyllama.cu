#include "cpu_backend.h"

CPUBackend::CPUBackend(
    const ModelConfig& config, 
    const ModelWeights& weights
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
    )
{}    

void CPUBackend::token_embedding(int token_id) {
    if (token_id < 0 || token_id >= config_.vocab_size) {
        throw std::out_of_range("token_id out of range");
    }

    const auto& embedding = weights_.embed_tokens;

    const std::size_t row_offset =
        static_cast<std::size_t>(token_id) *
        static_cast<std::size_t>(config_.hidden_size);

    std::ranges::copy_n(
        embedding.begin() + row_offset,
        config_.hidden_size,
        run_state_.x.begin()
    );
}

void CPUBackend::rmsnorm(const std::vector<float>& weight) {
    const int hidden_size = config_.hidden_size;

    const auto& x = run_state_.x;
    auto& out = run_state_.xb;  

    // sum(x^2)
    float sum = 0.0f;
    for (int i = 0; i < hidden_size; ++i) {
        sum += x[i] * x[i];
    }

    // 1 / sqrt(mean(x^2) + eps)
    const float scale =
        1.0f /
        std::sqrt(
            sum / static_cast<float>(hidden_size) + config_.rms_norm_eps
        );

    // x * scale * weight
    for (int i = 0; i < hidden_size; ++i) {
        out[i] = x[i] * scale * weight[i];
    }
}

void CPUBackend::qkv_projection(int layer) {
    const auto& attention = weights_.layers[layer].self_attn;

    // Q
    // xb [2048] -> Wq [2048, 2048] -> q [2048]
    linear(attention.q_proj, run_state_.xb, run_state_.q);

    // K
    // xb [2048] -> Wk [256, 2048] -> k [256]
    linear(attention.k_proj, run_state_.xb, run_state_.k);

    // V
    // xb [2048] -> Wv [256, 2048] -> v [256]
    linear(attention.v_proj, run_state_.xb, run_state_.v);
}

void CPUBackend::apply_rope(int position) {  
    const int head_dim = config_.hidden_size / config_.n_heads;

    const int num_heads    = config_.n_heads;
    const int num_kv_heads = config_.n_kv_heads;

    const float theta = config_.rope_theta;

    // Llama 的 rotate_half：前半部與後半部配對
    const int half = head_dim / 2;

    // Q : [32, 64]
    for (int head = 0; head < num_heads; ++head) {
        const int offset = head * head_dim;

        for (int i = 0; i < half; ++i) {
            // 頻率索引：0~31
            const float exponent =
                static_cast<float>(2 * i) / static_cast<float>(head_dim);

            const float inv_freq = 1.0f / std::pow(theta, exponent);

            const float angle = static_cast<float>(position) * inv_freq;

            const float c = std::cos(angle);
            const float s = std::sin(angle);

            // (i, i+32)
            const float x0 = run_state_.q[offset + i];
            const float x1 = run_state_.q[offset + i + half];

            run_state_.q[offset + i]        = x0 * c - x1 * s;
            run_state_.q[offset + i + half] = x1 * c + x0 * s;
        }
    }

    // K : [4, 64]
    for (int head = 0; head < num_kv_heads; ++head) {
        const int offset = head * head_dim;

        for (int i = 0; i < half; ++i) {
            const float exponent =
                static_cast<float>(2 * i) / static_cast<float>(head_dim);

            const float inv_freq = 1.0f / std::pow(theta, exponent);

            const float angle = static_cast<float>(position) * inv_freq;

            const float c = std::cos(angle);
            const float s = std::sin(angle);

            const float x0 = run_state_.k[offset + i];
            const float x1 = run_state_.k[offset + i + half];

            run_state_.k[offset + i]        = x0 * c - x1 * s;
            run_state_.k[offset + i + half] = x1 * c + x0 * s;
        }
    }
}

void CPUBackend::store_kv_cache(int layer, int position) {
    const int head_dim = config_.hidden_size / config_.n_heads;

    const int kv_dim = config_.n_kv_heads * head_dim;

    const size_t layer_offset =
        static_cast<size_t>(layer) * config_.max_seq_len * kv_dim;

    const size_t position_offset = 
        static_cast<size_t>(position) * kv_dim;

    const size_t offset = layer_offset + position_offset;

    for (int i = 0; i < kv_dim; ++i) {
        run_state_.key_cache[offset + i] = run_state_.k[i];
        run_state_.value_cache[offset + i] = run_state_.v[i];
    }
}

void CPUBackend::attention(int layer, int position)   {
    attention_score(layer, position);
    attention_softmax(position);
    attention_weighted_sum(layer, position);
    attention_output(layer);  
}

void CPUBackend::attention_score(int layer, int position) {
    const int num_heads    = config_.n_heads;
    const int num_kv_heads = config_.n_kv_heads;

    const int head_dim = config_.hidden_size / num_heads;

    const int seq_len = config_.max_seq_len;

    const int kv_dim = num_kv_heads * head_dim;

    // ------------------------------------------------------------
    // GQA
    //
    // TinyLlama:
    //
    // Q heads  = 32
    // KV heads = 4
    //
    // 每個 KV head 被 8 個 Q head 共用
    //
    // Q0  ~ Q7   -> KV0
    // Q8  ~ Q15  -> KV1
    // Q16 ~ Q23  -> KV2
    // Q24 ~ Q31  -> KV3
    // ------------------------------------------------------------
    const int heads_per_kv = num_heads / num_kv_heads;

    // ------------------------------------------------------------
    // 目前 layer 在 KV cache 中的起始位置
    //
    // KV cache layout:
    // [layer][position][kv_head][head_dim]
    //
    // 實際上一維：layer * max_seq_len * kv_dim
    // ------------------------------------------------------------

    const std::size_t layer_offset =
        static_cast<std::size_t>(layer) *
        static_cast<std::size_t>(seq_len) *
        static_cast<std::size_t>(kv_dim);

    // ============================================================
    // 遍歷 Query Heads
    // ============================================================
    for (int q_head = 0; q_head < num_heads; ++q_head) {
        // 找到這個 Q head 對應的 KV head
        const int kv_head = q_head / heads_per_kv;

        // Q offset
        // q layout: [q_head][head_dim]
        const std::size_t q_offset =
            static_cast<std::size_t>(q_head) *
            static_cast<std::size_t>(head_dim);

        // KV head 在一個 position 裡面的 offset
        const std::size_t kv_head_offset =
            static_cast<std::size_t>(kv_head) *
            static_cast<std::size_t>(head_dim);

        // 遍歷目前 token 以及之前所有 token (Causal Attention)
        for (int pos = 0; pos <= position; ++pos) {
            // ----------------------------------------------------
            // KV cache position offset
            // [layer][position][kv_head][head_dim]
            // ----------------------------------------------------
            const std::size_t position_offset =
                static_cast<std::size_t>(pos) *
                static_cast<std::size_t>(kv_dim);

            const std::size_t k_offset =
                layer_offset + position_offset + kv_head_offset;

            // Q · K
            float score = 0.0f;
            for (int i = 0; i < head_dim; ++i) {
                score +=
                    run_state_.q[q_offset + i] *
                    run_state_.key_cache[k_offset + i];
            }
          
            // score = (Q · K) / sqrt(head_dim)
            // TinyLlama: head_dim = 64, sqrt(64) = 8
            score /= std::sqrt(static_cast<float>(head_dim));

            // Attention score layout: [q_head][position]
            // att size: num_heads * seq_len
            const std::size_t att_offset =
                static_cast<std::size_t>(q_head) *
                static_cast<std::size_t>(seq_len) +
                static_cast<std::size_t>(pos);

            run_state_.att[att_offset] = score;
        }
    }
}

void CPUBackend::attention_softmax(int position) {
    for (int head = 0; head < config_.n_heads; ++head) {
        float* scores = 
            run_state_.att.data() + head * config_.max_seq_len;
        
        softmax(scores, position + 1);
    }
}

void CPUBackend::attention_weighted_sum(int layer, int position) {
    const int num_heads = config_.n_heads;
    const int num_kv_heads = config_.n_kv_heads;

    const int head_dim = config_.hidden_size / config_.n_heads;

    const int kv_dim = num_kv_heads * head_dim;

    //const int current_position = position_;

    // ---------------------------------------------------------
    // Attention output
    //
    // xb2 = [hidden_size] = [2048]
    //
    // layout:
    // [head0 64]
    // [head1 64]
    // ...
    // [head31 64]
    // ---------------------------------------------------------
    std::fill(run_state_.xb2.begin(), run_state_.xb2.end(), 0.0f);

    // ---------------------------------------------------------
    // Layer offset
    // KV Cache layout: [layer][position][kv_dim]
    // ---------------------------------------------------------
    const size_t layer_offset = 
        static_cast<size_t>(layer) * config_.max_seq_len * kv_dim;

    // ---------------------------------------------------------
    // 每一個 Query Head
    // ---------------------------------------------------------
    for (int head = 0; head < num_heads; ++head) {
        // -----------------------------------------------------
        // GQA: 32 Q heads, 4 KV heads
        // Q 0~7    -> KV 0
        // Q 8~15   -> KV 1
        // Q 16~23  -> KV 2
        // Q 24~31  -> KV 3
        // -----------------------------------------------------
        const int kv_head = head / (num_heads / num_kv_heads);

        // -----------------------------------------------------
        // Output location
        // head 0 -> xb2[0 ... 63]
        // head 1 -> xb2[64 ... 127]
        // ...
        // -----------------------------------------------------
        float* output = run_state_.xb2.data() + head * head_dim;

        // -----------------------------------------------------
        // Attention probability
        // att[head][position]
        // -----------------------------------------------------  
        const float* att = 
            run_state_.att.data() + head * config_.max_seq_len;

        // -----------------------------------------------------
        // 遍歷所有可以看到的 token (Causal Attention)
        // -----------------------------------------------------
        for (int pos = 0; pos <= position; ++pos) {
            const float attention_weight = att[pos];

            // -------------------------------------------------
            // Position offset
            // [layer][position][kv_dim]
            // -------------------------------------------------
            const size_t position_offset =
                static_cast<size_t>(pos) * kv_dim;

            // -------------------------------------------------
            // KV head offset
            //
            // KV head 0 -> 0
            // KV head 1 -> 64
            // KV head 2 -> 128
            // KV head 3 -> 192
            // -------------------------------------------------
            const size_t kv_head_offset =
                static_cast<size_t>(kv_head) * head_dim;

            // -------------------------------------------------
            // 完整 V Cache offset
            // layer + position + kv head
            // -------------------------------------------------
            const size_t value_offset =
                layer_offset + position_offset + kv_head_offset;

            const float* value =
                run_state_.value_cache.data() + value_offset;

            // -------------------------------------------------
            // Weighted Sum
            // output[i] += attention_weight * V[i]
            // -------------------------------------------------
            for (int i = 0; i < head_dim; ++i) {
                output[i] += attention_weight * value[i];
            }
        }
    }
}  

void CPUBackend::attention_output(int layer) {
    linear(
        weights_.layers[layer].self_attn.o_proj,
        run_state_.xb2,
        run_state_.xb
    );
}

void CPUBackend::residual_add() {
    const int hidden = config_.hidden_size;

    for (int i = 0; i < hidden; ++i) {
        run_state_.x[i] += run_state_.xb[i];
    }
}

void CPUBackend::final_rmsnorm() { 
    //rmsnorm(weights_.norm, run_state_.x, run_state_.xb); 
    rmsnorm(weights_.norm);
}

void CPUBackend::mlp(int layer) {
    // Gate
    linear(
        weights_.layers[layer].mlp.gate_proj,
        run_state_.xb,
        run_state_.hb
    );

    // Up
    linear(
        weights_.layers[layer].mlp.up_proj, 
        run_state_.xb, 
        run_state_.hb2
    );

    // SiLU(gate) * up
    silu_and_multiply();

    // Down
    linear(
        weights_.layers[layer].mlp.down_proj,
        run_state_.hb,
        run_state_.xb
    );
}

void CPUBackend::compute_logits() {
    const int vocab_size  = config_.vocab_size;
    const int hidden_size = config_.hidden_size;

    // ------------------------------------------------------------
    // LM Head
    // lm_head: [vocab_size, hidden_size]
    // TinyLlama: [32000, 2048]
    // ------------------------------------------------------------

    const std::vector<float>& lm_head = weights_.lm_head;

    const std::vector<float>& x = run_state_.xb;

    std::vector<float>& logits = run_state_.logits;

    // ------------------------------------------------------------
    // Size check
    // ------------------------------------------------------------

    if (x.size() != static_cast<size_t>(hidden_size)) {
        throw std::runtime_error(
            "compute_logits: hidden state size mismatch."
        );
    }

    const size_t expected_lm_head_size =
        static_cast<size_t>(vocab_size) *
        static_cast<size_t>(hidden_size);

    if (lm_head.size() != expected_lm_head_size) {
        throw std::runtime_error(
            "compute_logits: LM head size mismatch."
        );
    }

    if (logits.size() != static_cast<size_t>(vocab_size)) {
        logits.resize(vocab_size);
    }

    // ------------------------------------------------------------
    // Matrix-Vector Multiplication
    //
    // lm_head:
    //
    //             hidden_size
    //          ┌───────────────┐
    // token 0  │               │
    // token 1  │               │
    // token 2  │               │
    //   ...    │               │
    // token N  │               │
    //          └───────────────┘
    //            vocab_size
    //
    // x:[hidden_size]
    // logits: [vocab_size]
    // logits[i] = sum_j lm_head[i][j] * x[j]
    // ------------------------------------------------------------

    for (int i = 0; i < vocab_size; ++i) {
        const size_t row_offset =
            static_cast<size_t>(i) *
            static_cast<size_t>(hidden_size);

        float sum = 0.0f;

        for (int j = 0; j < hidden_size; ++j) {
            const float weight = lm_head[row_offset + j];
            sum += weight * x[j];
        }

        logits[i] = sum;
    }
}

void CPUBackend::linear(  
    const std::vector<float>& weight,
    const std::vector<float>& input,
    std::vector<float>& output
)
{
    const int input_size = static_cast<int>(input.size());
    const int output_size = static_cast<int>(output.size());

    // ------------------------------------------------------------
    // W 的元素數量必須是： output_size * input_size
    // W: [output_size, input_size]
    // ------------------------------------------------------------
    // 這也許不需要?
    const std::size_t expected_weight_size =
        static_cast<std::size_t>(output_size) *
        static_cast<std::size_t>(input_size);

    if (weight.size() != expected_weight_size) {
        throw std::runtime_error(
            "Linear weight size mismatch"
        );
    }

    // ------------------------------------------------------------
    // Matrix-vector multiplication
    // output[i] = sum_j W[i][j] * input[j]
    // ------------------------------------------------------------
    for (int i = 0; i < output_size; ++i) {
        float sum = 0.0f;

        const std::size_t row_offset =
            static_cast<std::size_t>(i) *
            static_cast<std::size_t>(input_size);

        for (int j = 0; j < input_size; ++j) {
            sum += weight[row_offset + j] * input[j];
        }

        output[i] = sum;
    }
}  

void CPUBackend::softmax(float* x, int size) {
    // 1. 找最大值
    float max_value = x[0];
    for (int i = 1; i < size; ++i) {
        max_value = std::max(max_value, x[i]);
    }

    // 2. exp(x - max)
    float sum = 0.0f;
    for (int i = 0; i < size; ++i) {
        x[i] = std::exp(x[i] - max_value);
        sum += x[i];
    }

    // 3. normalize
    const float inv_sum = 1.0f / sum;
    for (int i = 0; i < size; ++i) {
        x[i] *= inv_sum;
    }
}

void CPUBackend::silu_and_multiply() {
    const int size = config_.intermediate_size;

    for (int i = 0; i < size; ++i) {
        const float gate = run_state_.hb[i];

        const float up = run_state_.hb2[i];

        // sigmoid(gate)
        const float sigmoid = 1.0f / (1.0f + std::exp(-gate));

        // SiLU(gate)
        const float silu = gate * sigmoid;

        // SiLU(gate) * up
        run_state_.hb[i] = silu * up;
    }
}

