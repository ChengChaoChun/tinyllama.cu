#include "gpu_transformer.h"

#include <algorithm>
#include <stdexcept>

GPUTransformer::GPUTransformer(
    const ModelConfig& config,
    const GPUModelWeights& weights
) : config_(config),
    weights_(weights),
    backend_(config, weights)
{}

void GPUTransformer::Prefill(const std::vector<int>& tokens) {
    if (tokens.empty()) return;

    const int seq_len = static_cast<int>(tokens.size());

    // Embed the entire prompt in parallel.
    backend_.TokenEmbedding(tokens);

    //Forward(0, seq_len, seq_len);
    Forward(position_, seq_len, position_ + seq_len); 

    position_ += seq_len;
}

void GPUTransformer::Decode(int token_id) {
    backend_.TokenEmbedding(token_id);

    constexpr int seq_len = 1;

    Forward(position_, seq_len, position_ + 1);

    ++position_;
}

void GPUTransformer::Forward(
    int start_position, int seq_len, int kv_seq_len
) {
    for (int layer = 0; layer < config_.n_layers; ++layer) {
        const auto& layer_weights = weights_.layers[layer];

        // Attention
        backend_.RmsNorm(layer_weights.input_layernorm, seq_len);  
        backend_.QkvProjection(layer, seq_len);
        backend_.ApplyRoPE(start_position, seq_len);
        backend_.StoreKvCache(layer, start_position, seq_len);
        backend_.Attention(layer, start_position, seq_len, kv_seq_len);
        backend_.ResidualAdd(seq_len);
        backend_.RmsNorm(layer_weights.post_attention_layernorm, seq_len);
        backend_.Mlp(layer, seq_len);  
        backend_.ResidualAdd(seq_len);
    }

    backend_.RmsNorm(weights_.norm, seq_len);  
    backend_.ComputeLogits(seq_len);
}

void GPUTransformer::Reset() {
    position_ = 0;
    backend_.ResetState();   
}

void GPUTransformer::Synchronize() const {
    backend_.Synchronize();  
}  

std::vector<float> GPUTransformer::Logits(int seq_len) const {
    return backend_.FetchLogits(seq_len);
}

int GPUTransformer::ArgmaxToken(int seq_len) const {
    const auto values = backend_.FetchLogits(seq_len);
  
    if (values.empty()) {
        throw std::runtime_error(
            "GPUTransformer::ArgmaxToken: logits are empty."
        );
    }

    const int vocab_size = config_.vocab_size;

    const float* last_logits =
        values.data()
        + static_cast<std::size_t>(seq_len - 1) * vocab_size;

    return static_cast<int>(
        std::max_element(
            last_logits,
            last_logits + vocab_size
        ) - last_logits
    );
}