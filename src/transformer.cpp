#include "transformer.h"

#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <cstring>
#include <cuda_runtime.h>

Transformer::Transformer(
    const ModelConfig& config,
    const ModelWeights& weights
) : config_(config), 
    backend_(config, weights)
{}  

void Transformer::prefill(const std::vector<int>& tokens) {
    if (tokens.empty()) return;

    if (tokens.size() > static_cast<size_t>(config_.max_seq_len)) {
        throw std::runtime_error(
            "Prompt is longer than max sequence length."
        );
    }

    // Process prompt tokens
    for (int token_id : tokens) {
        forward_token(token_id);
    }
}

void Transformer::decode(int token_id) {
    forward_token(token_id);
}

void Transformer::forward_token(int token_id){
    // 1. Token Embedding
    //token_embedding(token_id);
    backend_.token_embedding(token_id);

    // 2. Transformer layers
    for (int layer = 0; layer < config_.n_layers; ++layer) {
        const auto& layer_weights = backend_.layer_weights(layer);

        // Input RMSNorm
        backend_.rmsnorm(layer_weights.input_layernorm);
        
        backend_.qkv_projection(layer);  

        backend_.apply_rope(position_);  

        backend_.store_kv_cache(layer, position_);  

        backend_.attention(layer, position_);  

        backend_.residual_add();

        // Post Attention RMSNorm  
        backend_.rmsnorm(layer_weights.post_attention_layernorm);

        backend_.mlp(layer);

        backend_.residual_add();
    }
    
    // Final RMSNorm
    backend_.final_rmsnorm();

    // LM Head
    backend_.compute_logits();

    // Move to next position
    ++position_;
}
   
int Transformer::argmax_token() const {
    const auto& logits = backend_.logits();

    int best_token = 0;
    float best_logit = logits[0];

    for (int i = 1; i < config_.vocab_size; ++i) {
        if (logits[i] > best_logit) {
            best_logit = logits[i];
            best_token = i;
        }
    }

    return best_token;
}
