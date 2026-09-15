#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <cstdint>
#include <cstddef>
#include <algorithm>
#include <limits>

#include "model_config.h"
#include "tokenizer.h"

#include "gpu_model_loader.h"
#include "gpu_transformer.h"  

// ============================================================
// Save float32 tensor to NPY
// ============================================================

void save_npy_float32(
    const std::string& filepath,
    const float* data,
    const std::vector<size_t>& shape
) {
    std::filesystem::path path(filepath);

    if (path.has_parent_path()) {
        std::filesystem::create_directories(path.parent_path());
    }

    std::ofstream out(filepath, std::ios::binary);

    if (!out.is_open()) {
        throw std::runtime_error(
            "無法建立/開啟檔案: " + filepath
        );
    }

    size_t total_elements = 1;

    for (size_t dim : shape) {
        total_elements *= dim;
    }

    std::string shape_str = "(";

    for (size_t i = 0; i < shape.size(); ++i) {
        shape_str += std::to_string(shape[i]);

        if (shape.size() == 1) {
            shape_str += ",";
        }
        else if (i + 1 < shape.size()) {
            shape_str += ", ";
        }
    }

    shape_str += ")";

    const char magic[] = "\x93NUMPY\x01\x00";

    out.write(magic, 8);

    std::string header_dict =
        "{'descr': '<f4', "
        "'fortran_order': False, "
        "'shape': " + shape_str + ", }";

    size_t current_len =
        8 + 2 + header_dict.length() + 1;

    size_t padding_len =
        (64 - (current_len % 64)) % 64;

    header_dict.append(padding_len, ' ');
    header_dict += '\n';

    uint16_t header_len =
        static_cast<uint16_t>(header_dict.length());

    out.write(
        reinterpret_cast<const char*>(&header_len),
        sizeof(header_len)
    );

    out.write(
        header_dict.c_str(),
        header_dict.length()
    );

    out.write(
        reinterpret_cast<const char*>(data),
        total_elements * sizeof(float)
    );
}


// ============================================================
// BF16 -> FP32
// ============================================================

std::vector<float> bf16_to_f32(
    const std::vector<__nv_bfloat16>& src
) {
    std::vector<float> dst(src.size());

    for (size_t i = 0; i < src.size(); ++i) {
        dst[i] = __bfloat162float(src[i]);
    }

    return dst;
}


// ============================================================
// UTF-8 Stream Decoder
//
// 用來避免 tokenizer decode 後中文 token
// 剛好切在 UTF-8 character 中間造成亂碼。
// ============================================================

class Utf8StreamDecoder {
private:
    std::vector<uint8_t> buffer;

    int get_utf8_char_len(uint8_t byte) const {
        if ((byte & 0x80) == 0x00)
            return 1;

        if ((byte & 0xE0) == 0xC0)
            return 2;

        if ((byte & 0xF0) == 0xE0)
            return 3;

        if ((byte & 0xF8) == 0xF0)
            return 4;

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

            if (processed_pos + expected_len <= buffer.size()) {
                std::cout.write(
                    reinterpret_cast<const char*>(&buffer[processed_pos]),
                    expected_len
                );

                processed_pos += expected_len;
            }
            else {
                // UTF-8 character 尚未完整
                break;
            }
        }

        buffer.erase(buffer.begin(), buffer.begin() + processed_pos);

        std::cout.flush();  
    }
};

int main() {
    try {
        ModelConfig config("models/TinyLlama-1.1B-Chat-v1.0/config.json");
        GPUModelLoader gpu_loader(
            "models/TinyLlama-1.1B-Chat-v1.0/model.safetensors", config
        );
        GPUModelWeights gpu_weights = gpu_loader.Load();
  
        // Tokenizer
        Tokenizer tokenizer(
            "models/TinyLlama-1.1B-Chat-v1.0/tokenizer.model"
        );  

        // GPU Transformer
        GPUTransformer transformer(config, gpu_weights);

        // ====================================================
        // Prompt
        // ====================================================
        const std::string prompt = "請詳細說明在 CUDA C++ 中實現高效能 GEMM (General Matrix Multiplication) 的核心優化技巧，包括 Shared Memory Tiling、Thread Coarsening 與 Memory Coalescing：";

        std::vector<int> tokens = tokenizer.encode(prompt, true, false);

        const int prompt_len = static_cast<int>(tokens.size());

        std::cout
            << "============================================\n"
            << " Prompt\n"
            << "============================================\n";

        std::cout << "Prompt: " << prompt << '\n';

        std::cout << "Tokens: ";
        for (int token : tokens) {
            std::cout << token << ' ';
        }

        std::cout << "\nPrompt length: " << prompt_len << "\n\n";

        // ====================================================
        // Prefill
        // ====================================================
        std::cout
            << "============================================\n"
            << " PREFILL\n"
            << "============================================\n";

        transformer.Prefill(tokens);
        transformer.Synchronize();

        // ====================================================
        // Fetch and save last prefill logits
        //
        // Shape: [1, vocab_size]
        // ====================================================
        const std::vector<float> prefill_logits =
            transformer.Logits(prompt_len);

        // Only the last position is used to generate the first token.
        const float* last_prefill_logits =
            prefill_logits.data()
            + static_cast<std::size_t>(prompt_len - 1)
            * config.vocab_size;

        save_npy_float32(
            "test/dumps/cpp_prefill_logits.npy",
            last_prefill_logits,
            {
                1,
                static_cast<size_t>(config.vocab_size)
            }
        );

        // ====================================================
        // First generated token
        // ====================================================
        int next_token = transformer.ArgmaxToken();
        std::cout << "\nPrefill next token: " << next_token << '\n';

        // ====================================================
        // Decode
        // ====================================================
        std::cout
            << "\n============================================\n"
            << " DECODE\n"
            << "============================================\n\n";

        std::cout << "Generated Result:\n";

        std::cout << prompt << std::flush;

        constexpr int max_new_tokens = 100;  
        constexpr int eos_token_id = 2;

        Utf8StreamDecoder stream_decoder;

        std::vector<int> generated_tokens;
        generated_tokens.reserve(max_new_tokens);

        // ====================================================
        // Decode loop
        // ====================================================
        for (int step = 0; step < max_new_tokens; ++step) {
            if (next_token == eos_token_id) {
                std::cout << "\n[EOS reached]\n";
                break;
            }

            // Token output
            std::string token_text =
                tokenizer.decode(
                    tokens.empty() ? -1 : tokens.back(), next_token
                );

            stream_decoder.print_bytes(token_text);

            generated_tokens.push_back(next_token);

            // ------------------------------------------------
            // Decode
            // ------------------------------------------------
            const int position = prompt_len + step;

            std::cout
                << "\n[step " << step
                << "] token=" << next_token
                << " position=" << position << '\n';

            transformer.Decode(next_token);

            transformer.Synchronize();

            // ------------------------------------------------
            // Fetch and save decode logits
            //
            // Shape: [1, vocab_size]
            // ------------------------------------------------
            const std::vector<float> decode_logits = 
                transformer.Logits(1);

            const std::string dump_path =
                "test/dumps/cpp_decode_logits_"
                + std::to_string(step) + ".npy";

            save_npy_float32(
                dump_path,
                decode_logits.data(),
                {
                    1,
                    static_cast<size_t>(config.vocab_size)
                }
            );

            // Next token
            next_token = transformer.ArgmaxToken();
        }

        // ====================================================
        // Generation Result
        // ====================================================
        std::cout
            << "\n\n============================================\n"
            << " Generation Finished\n"
            << "============================================\n";

        std::cout
            << "Generated tokens: " << generated_tokens.size() << '\n';

        std::cout << "Token IDs:\n";

        for (int token : generated_tokens) {
            std::cout << token << ' ';
        }

        std::cout << "\n";

        // ====================================================
        // Save generated token IDs
        // ====================================================
        {
            std::ofstream token_out("test/dumps/cpp_tokens.txt");

            if (!token_out.is_open()) {
                throw std::runtime_error(
                    "無法建立 cpp_tokens.txt"
                );
            }

            for (int token : generated_tokens) {
                token_out << token << '\n';
            }
        }

        // ====================================================
        // Result
        // ====================================================

        std::cout
            << "\nDump files saved to:\n"
            << "  test/dumps/cpp_prefill_logits.npy\n"
            << "  test/dumps/cpp_decode_logits_*.npy\n"
            << "  test/dumps/cpp_tokens.txt\n";

        std::cout
            << "\n============================================\n"
            << " CUDA Inference Complete\n"
            << "============================================\n";
    }
    catch (const std::exception& e) {
        std::cerr << "\nError: " << e.what() << '\n';
        return 1;
    }  

    return 0;
}
