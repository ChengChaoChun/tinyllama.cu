#pragma once

#include <vector>

#include "model_config.h"
#include "gpu_weights.cuh"  
#include "backend/gpu_backend.h"

class GPUTransformer {
public:
    GPUTransformer(
        const ModelConfig& config,
        const GPUModelWeights& weights
    );

    ~GPUTransformer() = default;

    GPUTransformer(const GPUTransformer&) = delete;
    GPUTransformer& operator=(const GPUTransformer&) = delete;

    GPUTransformer(GPUTransformer&&) noexcept = default;
    GPUTransformer& operator=(GPUTransformer&&) noexcept = delete;

    // Prefill the entire prompt.
    void Prefill(const std::vector<int>& tokens);

    // Decode one token.
    void Decode(int token_id);

    // Reset the inference state.
    void Reset();

    // Synchronize the GPU backend.
    void Synchronize() const;

    // Fetch the current logits from the GPU.
    std::vector<float> Logits(int seq_len) const;  

    // Return the token with the highest logit.
    int ArgmaxToken(int seq_len);  

    const int Position() { return position_; }  

private:
    const ModelConfig& config_;
    const GPUModelWeights& weights_;

    GPUBackend backend_;

    // Position of the next token in the sequence.
    int position_ = 0;

private:
    // Run all transformer layers for the current sequence.
    void Forward(int start_position, int seq_len, int kv_seq_len);
};