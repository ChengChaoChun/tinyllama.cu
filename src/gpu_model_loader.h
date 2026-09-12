#pragma once

#include <string>

#include "gpu_weights.cuh"
#include "safetensor_reader.h"
#include "model_config.h"

class GPUModelLoader {
public:
    GPUModelLoader(
        const std::string& model_path,
        const ModelConfig& config
    );

    // safetensors BF16 → GPUModelWeights
    GPUModelWeights Load();
  
private:
    // TensorView(BF16) → DeviceBuffer<BF16>
    DeviceBuffer<__nv_bfloat16> LoadTensor(
        const std::string& name
    );

    GPULayerWeights LoadLayer(int layer_id);

private:
    SafeTensorReader reader_;

    const ModelConfig& config_;
};