#include <iostream>
#include <string>
#include <vector>
#include <exception>
#include <filesystem>  

#include "tokenizer.h"
#include "model_config.h"
#include "cpu_model_loader.h"
#include "transformer.h"

// 將 1D float32 陣列儲存為 NumPy .npy (v1.0 格式)
void save_logits_npy(const std::string& filepath, const float* data, size_t size) {
    // 自動建立父資料夾 (例如 test/dumps/)
    std::filesystem::path path(filepath);
    if (path.has_parent_path()) {
        std::filesystem::create_directories(path.parent_path());
    }

    std::ofstream out(filepath, std::ios::binary);
    if (!out.is_open()) {
        throw std::runtime_error("無法建立/開啟檔案: " + filepath);
    }

    // 1. NPY Magic Header (\x93NUMPY) + Version 1.0 (\x01\x00)
    const char magic[] = "\x93NUMPY\x01\x00";
    out.write(magic, 8);

    // 2. 建立 HeaderDict (描述數據類型 <f4 為 little-endian float32、形狀與連續性)
    std::string header_dict = "{'descr': '<f4', 'fortran_order': False, 'shape': (" 
                            + std::to_string(size) + ",), }";

    // 3. 按照 NumPy NPY v1.0 規範進行 64-byte 對齊 padding
    // 計算公式：MAGIC(8) + HEADER_LEN_SIZE(2) + header_dict.length() + padding + '\n' == 64 的倍數
    size_t current_len = 8 + 2 + header_dict.length() + 1; // +1 是為了最後的 '\n'
    size_t padding_len = (64 - (current_len % 64)) % 64;
    
    header_dict.append(padding_len, ' ');
    header_dict += '\n';

    // 4. 寫入 Header 長度 (2-byte unsigned short, Little-Endian)
    uint16_t header_len = static_cast<uint16_t>(header_dict.length());
    out.write(reinterpret_cast<const char*>(&header_len), sizeof(header_len));

    // 5. 寫入 Header 字串與二進位數據
    out.write(header_dict.c_str(), header_dict.length());
    out.write(reinterpret_cast<const char*>(data), size * sizeof(float));
}

// -----------------------------------------------------------------------------
// UTF-8 流式解碼器 (解決中文 Byte-level BPE Split 導致亂碼的問題)
// -----------------------------------------------------------------------------
class Utf8StreamDecoder {
private:
    std::vector<uint8_t> buffer;

    // 根據 UTF-8 Header Byte 判斷字元總長度
    int get_utf8_char_len(uint8_t byte) const {
        if ((byte & 0x80) == 0x00) return 1;        // ASCII (0xxxxxxx)
        if ((byte & 0xE0) == 0xC0) return 2;        // 2-bytes (110xxxxx)
        if ((byte & 0xF0) == 0xE0) return 3;        // 3-bytes (1110xxxx, 常見中文)
        if ((byte & 0xF8) == 0xF0) return 4;        // 4-bytes (11110xxx)
        return 1; // 異常字元強制消耗 1 byte
    }

public:
    void print_bytes(const std::string& bytes_str) {
        // 將字串中的 raw bytes 塞入 buffer
        for (char c : bytes_str) {
            buffer.push_back(static_cast<uint8_t>(c));
        }

        size_t processed_pos = 0;
        while (processed_pos < buffer.size()) {
            uint8_t first_byte = buffer[processed_pos];
            int expected_len = get_utf8_char_len(first_byte);

            // 只有當 buffer 累積足夠長度能拼出完整 UTF-8 字元時才印出
            if (processed_pos + expected_len <= buffer.size()) {
                std::cout.write(reinterpret_cast<const char*>(&buffer[processed_pos]), expected_len);
                processed_pos += expected_len;
            } else {
                // 資料不足（例如 3-byte 中文只拿到 1~2 byte），暫緩印出，等待下一個 token
                break;
            }
        }

        // 清除已輸出的 byte
        buffer.erase(buffer.begin(), buffer.begin() + processed_pos);
        std::cout.flush();
    }
};

int main() {
    try {
        // 1. 載入模型配置與權重
        ModelConfig config("model/tinyllama-1.1b/config.json");  

        CPUModelLoader loader(
            "model/tinyllama-1.1b/model.safetensors", config
        );  
        
        ModelWeights weights = loader.load();
        
        Transformer model(config, std::move(weights));  

        // 2. 載入 Tokenizer
        Tokenizer tokenizer("model/tinyllama-1.1b/tokenizer.model");

        // 3. Prompt 設定與 Tokenize
        std::string prompt = "Hey, 你好";
        auto tokens = tokenizer.encode(prompt, true , false);  

        std::cout << "================================\n";
        std::cout << "Prompt: " << prompt << '\n';
        std::cout << "Token Count: " << tokens.size() << '\n';
        std::cout << "================================\n\n";

        // ============================================================
        // 4. Prefill Phase (建立 Prompt KV Cache 並 Dump Logits)
        // ============================================================
        model.prefill(tokens);    
        int next_token = model.argmax_token();
         
        const auto& prefill_logits = model.logits();
        save_logits_npy(
            "test/dumps/cpp_prefill_logits_last.npy",
            prefill_logits.data(),
            prefill_logits.size()
        );
        
        // ============================================================
        // 5. Autoregressive Decode Phase (自回歸生成與 Logits 記錄)
        // ============================================================
        std::cout << "Generated Result:\n";
        std::cout << prompt << std::flush; // 先印出 Prompt 原文

        const int max_new_tokens = 50;
        const int eos_token_id = 2; // LLaMA / TinyLlama EOS Token
        
        Utf8StreamDecoder stream_decoder;
        std::vector<int> generated_tokens;
        int prev_token = tokens.back(); 

        for (int step = 0; step < max_new_tokens; ++step) {
            if (next_token == eos_token_id) {
                std::cout << "\n[EOS reached]";
                break;
            }

            // A. Dump 當前 Step 計算出來的 Logits (對應要預測 next_token 的機率分佈)
            const auto& logits = model.logits();
            std::string dump_path = "test/dumps/cpp_decode_logits_" + std::to_string(step) + ".npy";
            save_logits_npy(dump_path, logits.data(), logits.size());

            // B. 流式解碼與輸出
            std::string token_str = tokenizer.decode(prev_token, next_token);
            stream_decoder.print_bytes(token_str);

            generated_tokens.push_back(next_token);

            // C. 推進 Transformer 狀態：輸入 next_token 計算下一個 token 的 logits
            prev_token = next_token;
            //model.forward(next_token);  
            model.decode(next_token);
            next_token = model.argmax_token();
        }

        // ============================================================
        // 6. 匯出生成的 Token ID 列表 (供 Python 腳本快速比對 ID 序列)
        // ============================================================
        std::ofstream token_out("test/dumps/cpp_tokens.txt");
        for (int id : generated_tokens) {
            token_out << id << "\n";
        }

        std::cout << "\n\n================================\n";
        std::cout << "Generation finished. Total generated tokens: " << generated_tokens.size() << '\n';
        std::cout << "Dumps saved to test/dumps/\n";
        std::cout << "================================\n";
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }

    return 0;
}