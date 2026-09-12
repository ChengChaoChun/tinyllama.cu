#pragma once

#include <string>

#include "cpu_weights.h"
#include "safetensor_reader.h"
#include "model_config.h"

class CPUModelLoader {
public:
    CPUModelLoader(
        const std::string& model_path,
        const ModelConfig& config
    );

    // safetensors BF16 → FP32 → ModelWeights 
    ModelWeights load();

private:
    // SafeTensorReader
    // 負責：
    // mmap
    // header
    // TensorView
    SafeTensorReader reader_;

    const ModelConfig& config_;

private:
    // TensorView -> BF16 → FP32 -> vector<float>
    std::vector<float> load_tensor(const std::string& name);

    LayerWeights load_layer(int layer_id);

    static std::vector<float> bf16_to_fp32(
        std::span<const std::byte> data
    );
};