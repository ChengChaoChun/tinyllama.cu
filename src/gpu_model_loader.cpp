#include "gpu_model_loader.h"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

GPUModelLoader::GPUModelLoader(
    const std::string& model_path,
    const ModelConfig& config
) : reader_(model_path),
    config_(config)
{}  

GPUModelWeights GPUModelLoader::Load() {
    GPUModelWeights weights;
  
    weights.embed_tokens = LoadTensor("model.embed_tokens.weight");

    // Transformer Layers
    weights.layers.reserve(config_.n_layers);
    for (int layer = 0; layer < config_.n_layers; ++layer) {
        weights.layers.push_back(LoadLayer(layer));
    }

    // Final RMSNorm [hidden_size]
    weights.norm = LoadTensor("model.norm.weight");

    // LM Head [vocab_size, hidden_size]
    weights.lm_head = LoadTensor("lm_head.weight");

    return weights;
}

DeviceBuffer<__nv_bfloat16> GPUModelLoader::LoadTensor(
    const std::string& name  
) {
    const TensorView view = reader_.view(name);

    // GPU ModelWeights 預期使用 BF16
    if (view.dtype != DType::BF16) {
        throw std::runtime_error("Expected BF16 tensor: " + name);
    }

    // BF16 = 2 bytes
    // 確認資料大小可以被 BF16 element size 整除
    constexpr std::size_t element_size = sizeof(__nv_bfloat16);
    if (view.bytes.size() % element_size != 0) {
        throw std::runtime_error(
            "Invalid BF16 tensor size: " + name
        );
    }

    // element count
    const std::size_t num_elements = view.bytes.size() / element_size;

    // --------------------------------------------------------
    // Host → Device
    // view.bytes: mmap file → std::span<const std::byte>
    // result: GPU memory
    // --------------------------------------------------------
    DeviceBuffer<__nv_bfloat16> result(num_elements);
    //result.copy_from_host(view.bytes.data(), num_elements);
    result.copy_from_host(
        reinterpret_cast<const __nv_bfloat16*>(view.bytes.data()),
        num_elements
    );

    return result;
}

GPULayerWeights GPUModelLoader::LoadLayer(int layer_id) {
    GPULayerWeights layer;

    const std::string prefix = 
        "model.layers." + std::to_string(layer_id);

    // Input RMSNorm
    layer.input_layernorm =
        LoadTensor(prefix + ".input_layernorm.weight");

    // Self Attention
    layer.self_attn.q_proj =
        LoadTensor(prefix + ".self_attn.q_proj.weight");

    layer.self_attn.k_proj =
        LoadTensor(prefix + ".self_attn.k_proj.weight");

    layer.self_attn.v_proj =
        LoadTensor(prefix + ".self_attn.v_proj.weight");

    layer.self_attn.o_proj =
        LoadTensor(prefix + ".self_attn.o_proj.weight");

    // Post Attention RMSNorm
    layer.post_attention_layernorm =
        LoadTensor(prefix + ".post_attention_layernorm.weight");

    // MLP
    layer.mlp.gate_proj =
        LoadTensor(prefix + ".mlp.gate_proj.weight");

    layer.mlp.up_proj =
        LoadTensor(prefix + ".mlp.up_proj.weight");

    layer.mlp.down_proj = 
        LoadTensor(prefix + ".mlp.down_proj.weight");


    return layer;
}