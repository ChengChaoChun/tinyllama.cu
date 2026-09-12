#pragma once
#include <iostream>
#include <fstream>
#include <string>
#include <stdexcept>  
#include <nlohmann/json.hpp>

// 1. 定義 Config 結構體，只抓推理（Inference）真正需要的欄位
class ModelConfig {  
public:
    explicit ModelConfig(const std::string& config_path) {
        LoadConfig(config_path);  
    }  

    int hidden_size;             // 2048
    int intermediate_size;       // 5632
    int n_layers;                // 22
    int n_heads;                 // 32
    int n_kv_heads;              // 4
    int vocab_size;              // 32000
    int max_seq_len;             // 2048
    
    // Hyperparameters
    float rms_norm_eps;          
    float rope_theta;            
    
    int bos_token_id;            
    int eos_token_id;            

private:
    void LoadConfig(const std::string& config_path) {
        std::ifstream file(config_path);
        if (!file.is_open()) {
            throw std::runtime_error(
                "Cannot open config file: " + config_path);
        }

        try {
            nlohmann::json j;
            file >> j;

            // Architecture
            hidden_size       = j.at("hidden_size").get<int>();
            intermediate_size = j.at("intermediate_size").get<int>();
            n_layers          = j.at("num_hidden_layers").get<int>();
            n_heads           = j.at("num_attention_heads").get<int>();
            n_kv_heads        = j.at("num_key_value_heads").get<int>();
            vocab_size        = j.at("vocab_size").get<int>();
            max_seq_len       = j.at("max_position_embeddings").get<int>();

            // Hyperparameters
            rms_norm_eps = j.at("rms_norm_eps").get<float>();
            rope_theta   = j.at("rope_theta").get<float>();

            // Special Tokens (有些模型可能沒有，所以給預設值)
            bos_token_id = j.value("bos_token_id", 1);
            eos_token_id = j.value("eos_token_id", 2);
        }
        catch (const nlohmann::json::exception& e) {
            throw std::runtime_error(
                std::string("Config JSON parse error: ") + e.what());
        }
    }  
};