#pragma once

#include "model_config.h"
#include "cpu_weights.h"
#include "backend/cpu_backend.h"  

class Transformer {
public:
    Transformer(
        const ModelConfig& config, 
        const ModelWeights& weights
    );  

    // Prefill: 一次處理整個 prompt
    void prefill(const std::vector<int>& tokens); 

    // Decode: 一次處理一個 token
    void decode(int token_id);  

    // 暫定放在 public
    int argmax_token() const;

    void reset() {
        position_ = 0;
        backend_.reset_state();
    }  

    // 暫時保留
    const std::vector<float>& logits() const {
        return backend_.logits();   
    }

private:
    const ModelConfig& config_;

    CPUBackend backend_;

    // 目前正在處理第幾個 token 
    // 第一次 forward: position_ = 0 
    // 第二次 forward: position_ = 1 
    // ... 
    int position_ = 0;

private:
    void forward_token(int token_id);
};