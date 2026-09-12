#include "cpu_model_loader.h"

#include <stdexcept>
#include <utility>
//#include <bit>  

CPUModelLoader::CPUModelLoader(
    const std::string& model_path,
    const ModelConfig& config
) : reader_(model_path),
    config_(config)
{}

// load 完整流程：
// safetensors → mmap → SafeTensorReader → TensorView 
// → CPUModelLoader → BF16 → FP32 → ModelWeights
ModelWeights CPUModelLoader::load() {
    ModelWeights weights;

    // Token Embedding : [vocab_size, hidden_size]
    // TinyLlama: [32000, 2048]
    weights.embed_tokens = load_tensor("model.embed_tokens.weight");

    // Transformer Layers
    weights.layers.reserve(config_.n_layers);
    for (int layer = 0; layer < config_.n_layers; ++layer) {
        weights.layers.push_back(load_layer(layer));  
    }

    // Final RMSNorm [hidden_size]
    weights.norm = load_tensor("model.norm.weight");

    // Language Model Head [vocab_size, hidden_size]
    weights.lm_head = load_tensor("lm_head.weight");

    return weights;
}

// TensorView(BF16) → FP32 → vector<float>
std::vector<float> CPUModelLoader::load_tensor(const std::string& name) {
    const TensorView view = reader_.view(name);

    // CPU ModelWeights 目前統一使用 FP32
    if (view.dtype != DType::BF16) {
        throw std::runtime_error("Expected BF16 tensor: " + name);
    }

    return bf16_to_fp32(view.bytes);
}

LayerWeights CPUModelLoader::load_layer(int layer_id) {
    LayerWeights layer;

    const std::string prefix = 
        "model.layers." + std::to_string(layer_id);

    // Input RMSNorm [hidden_size]
    layer.input_layernorm =
        load_tensor(prefix + ".input_layernorm.weight");

    // Self Attention
    layer.self_attn.q_proj =
        load_tensor(prefix + ".self_attn.q_proj.weight");

    layer.self_attn.k_proj =
        load_tensor(prefix + ".self_attn.k_proj.weight");

    layer.self_attn.v_proj =
        load_tensor(prefix + ".self_attn.v_proj.weight");

    layer.self_attn.o_proj =
        load_tensor(prefix + ".self_attn.o_proj.weight");

    // Post Attention RMSNorm
    layer.post_attention_layernorm =
        load_tensor(prefix + ".post_attention_layernorm.weight");

    // MLP
    layer.mlp.gate_proj =
        load_tensor(prefix + ".mlp.gate_proj.weight");

    layer.mlp.up_proj =
        load_tensor(prefix + ".mlp.up_proj.weight");

    layer.mlp.down_proj =
        load_tensor(prefix + ".mlp.down_proj.weight");

    return layer;
}

std::vector<float> CPUModelLoader::bf16_to_fp32(
    std::span<const std::byte> data
) {
    // BF16 = 2 bytes
    constexpr std::size_t bf16_bytes = sizeof(std::uint16_t);

    // BF16 data 大小必須是 2 的倍數
    if (data.size() % bf16_bytes != 0) {
        throw std::runtime_error(
            "Invalid BF16 data size: " +
            std::to_string(data.size())
        );
    }

    const std::size_t num_elements = data.size() / bf16_bytes;
    
    std::vector<float> result(num_elements);

    // BF16 → FP32
    for (std::size_t i = 0; i < num_elements; ++i) {
        std::uint16_t bf16_bits;

        std::memcpy(
            &bf16_bits, 
            data.data() + i * bf16_bytes,
            sizeof(bf16_bits)
        );

        const std::uint32_t fp32_bits =
            static_cast<std::uint32_t>(bf16_bits) << 16;  

        float value;
        std::memcpy(&value, &fp32_bits, sizeof(float));
        result[i] = value;  
        //result[i] = std::bit_cast<float>(fp32_bits);
    }

    return result;
}
