#pragma once

#include "../model_config.h"
#include "../cpu_weights.h"
#include "../run_state.h"

#include <cstddef>
#include <vector>

class CPUBackend {
public:
    CPUBackend(const ModelConfig& config, const ModelWeights& weights);

    const LayerWeights& layer_weights(int layer) const {
        return weights_.layers[layer];
    }

    // Reset runtime state.
    void reset_state() { run_state_.reset(); }; 

    //-------------- Transformer Operations --------------
    void token_embedding(int token_id);

    void rmsnorm(const std::vector<float>& weight);  

    void qkv_projection(int layer);

    void apply_rope(int position);

    void store_kv_cache(int layer, int position);

    void attention(int layer, int position);

    void residual_add();

    void mlp(int layer);

    void final_rmsnorm();

    void compute_logits();

    // Runtime State Access
    const RunState& state() const { return run_state_; }  
    const std::vector<float>& logits() const { return run_state_.logits; }

private:
    void linear(
        const std::vector<float>& weight,
        const std::vector<float>& input,
        std::vector<float>& output
    );

    void attention_score(int layer, int position);
    void attention_softmax(int position);  
    void attention_weighted_sum(int layer, int position);
    void attention_output(int layer);

    void softmax(float* x, int size);

    void silu_and_multiply();

private:
    // Model
    const ModelConfig& config_;
    const ModelWeights& weights_;

    RunState run_state_;
};