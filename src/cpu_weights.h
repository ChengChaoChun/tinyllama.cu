#pragma once
#include <vector>
#include "safetensor_reader.h"

struct AttentionWeights {
    std::vector<float> q_proj; // [hidden_size, hidden_size]
    std::vector<float> k_proj; // [num_key_value_heads * head_dim, hidden_size]
    std::vector<float> v_proj; // [num_key_value_heads * head_dim, hidden_size]
    std::vector<float> o_proj; // [hidden_size, hidden_size]
}; 

struct MLPWeights {
    std::vector<float> gate_proj; // [intermediate_size, hidden_size]
    std::vector<float> up_proj;   // [intermediate_size, hidden_size]
    std::vector<float> down_proj; // [hidden_size, intermediate_size]
};

// one Transformer Layer weights
struct LayerWeights {
    std::vector<float> input_layernorm;
    AttentionWeights self_attn;
    std::vector<float> post_attention_layernorm;
    MLPWeights mlp;
};
  
struct ModelWeights {  
    // --------------------------------------------------------
    // Token Embedding : [vocab_size, hidden_size]
    // TinyLlama : [32000, 2048]
    // --------------------------------------------------------
    std::vector<float> embed_tokens;

    // --------------------------------------------------------
    // Transformer Layers
    // TinyLlama: num_hidden_layers = 22
    // --------------------------------------------------------
    std::vector<LayerWeights> layers;

    // --------------------------------------------------------
    // last RMSNorm
    // [hidden_size]
    // TinyLlama: [2048]
    // --------------------------------------------------------
    std::vector<float> norm;  

    // --------------------------------------------------------
    // Language Model Head
    // [vocab_size, hidden_size]
    // TinyLlama:[32000, 2048]
    // 用來把 hidden state -> logits
    // --------------------------------------------------------
    std::vector<float> lm_head;  
};