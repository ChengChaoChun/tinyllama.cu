#include <iostream>
#include <string>
#include <vector>
#include <stdexcept>

#include "model_config.h"
#include "tokenizer.h"

#include "gpu_model_loader.h"
#include "gpu_transformer.h"


// ============================================================
// UTF-8 Stream Decoder
//
// 避免 tokenizer decode 後，中文 token 剛好切在
// UTF-8 character 中間造成亂碼。
// ============================================================
class Utf8StreamDecoder {
private:
    std::vector<uint8_t> buffer;

    int get_utf8_char_len(uint8_t byte) const {
        if ((byte & 0x80) == 0x00) return 1;
        if ((byte & 0xE0) == 0xC0) return 2;
        if ((byte & 0xF0) == 0xE0) return 3;
        if ((byte & 0xF8) == 0xF0) return 4;

        return 1;
    }

public:
    void print_bytes(const std::string& bytes_str) {
        for (char c : bytes_str) {
            buffer.push_back(static_cast<uint8_t>(c));
        }

        std::size_t processed_pos = 0;

        while (processed_pos < buffer.size()) {
            uint8_t first_byte = buffer[processed_pos];

            int expected_len = get_utf8_char_len(first_byte);

            // UTF-8 character 尚未完整，等待下一個 token
            if (processed_pos + expected_len > buffer.size()) {
                break;
            }

            std::cout.write(
                reinterpret_cast<const char*>(&buffer[processed_pos]),
                expected_len
            );

            processed_pos += expected_len;
        }

        // 移除已經輸出的 bytes
        buffer.erase(
            buffer.begin(),
            buffer.begin() + processed_pos
        );

        std::cout.flush();
    }

    void flush() {
        if (!buffer.empty()) {
            std::cout.write(
                reinterpret_cast<const char*>(buffer.data()),
                buffer.size()
            );

            buffer.clear();
            std::cout.flush();
        }
    }
};

int main() {
    try {
        // Model
        ModelConfig config(
            "models/TinyLlama-1.1B-Chat-v1.0/config.json"
        );

        GPUModelLoader gpu_loader(
            "models/TinyLlama-1.1B-Chat-v1.0/model.safetensors",
            config
        );

        GPUModelWeights gpu_weights = gpu_loader.Load();

        // Tokenizer
        Tokenizer tokenizer(
            "models/TinyLlama-1.1B-Chat-v1.0/tokenizer.model"
        );

        // GPU Transformer
        GPUTransformer transformer(config, gpu_weights);

        std::cout
            << "============================================\n"
            << " TinyLlama CUDA Inference Engine\n"
            << " Type 'exit' to quit.\n"
            << "============================================\n\n";

        constexpr int max_new_tokens = 100;
        constexpr int eos_token_id = 2;

        std::string prompt;

        while (true) {
            // User input
            std::cout << "> " << std::flush;

            if (!std::getline(std::cin, prompt)) {
                break;
            }

            if (prompt == "exit") {
                std::cout << "Bye!\n";
                break;
            }

            if (prompt.empty()) {
                continue;
            }


            // ------------------------------------------------
            // Reset previous conversation state
            //
            // 每一次輸入都是獨立的 inference。
            // 不保留上一輪的 KV cache。
            // ------------------------------------------------
            transformer.Reset();

            // Tokenize
            std::vector<int> tokens =
                tokenizer.encode(prompt, true, false);

            if (tokens.empty()) {
                std::cout << '\n';
                continue;
            }

            const int prompt_len = static_cast<int>(tokens.size());


            // Prefill
            transformer.Prefill(tokens);
            transformer.Synchronize();

            // First generated token
            int next_token = transformer.ArgmaxToken(prompt_len);

            // Decode
            Utf8StreamDecoder stream_decoder;

            std::cout << '\n';
  
            std::vector<int> generated_tokens;
            generated_tokens.reserve(max_new_tokens);

            for (int step = 0; step < max_new_tokens; ++step) {
                if (next_token == eos_token_id) break;

                // --------------------------------------------
                // Decode token
                //
                // 使用前一個 token 作為 decode context，
                // 讓 tokenizer 正確處理 token decode。
                // --------------------------------------------
                std::string token_text =
                    tokenizer.decode(
                        tokens.empty() ? -1 : tokens.back(),
                        next_token
                    );

                stream_decoder.print_bytes(token_text);

                generated_tokens.push_back(next_token);

                // Feed generated token back into model
                transformer.Decode(next_token);
                transformer.Synchronize();

                next_token = transformer.ArgmaxToken(1);
            }

            stream_decoder.flush();
            std::cout << "\n\n";
        }
    }
    catch (const std::exception& e) {
        std::cerr << "\nError: " << e.what() << '\n';
        return 1;
    }

    return 0;
}