#include <algorithm>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "model_config.h"
#include "gpu_model_loader.h"
#include "gpu_transformer.h"
#include "tokenizer.h"

// Save float32 array as .npy
void SaveNpyFloat32(
    const std::string& path,
    const float* data,
    const std::vector<std::size_t>& shape
) {
    std::ofstream out(path, std::ios::binary);

    if (!out.is_open()) {
        throw std::runtime_error("Failed to open output file: " + path);
    }

    // Build NumPy header
    std::string shape_str = "(";

    for (std::size_t i = 0; i < shape.size(); ++i) {
        shape_str += std::to_string(shape[i]);

        if (shape.size() == 1) {
            shape_str += ",";
        } else if (i + 1 < shape.size()) {
            shape_str += ", ";
        }
    }

    shape_str += ")";

    std::string header =
        "{'descr': '<f4', 'fortran_order': False, 'shape': "
        + shape_str
        + ", }";

    // NumPy .npy format version 1.0
    const std::size_t preamble_size = 10;
    const std::size_t padding =
        16 - ((preamble_size + header.size() + 1) % 16);

    header += std::string(padding, ' ');
    header += '\n';

    if (header.size() > 65535) {
        throw std::runtime_error("NumPy header is too large");
    }

    const char magic[] = "\x93NUMPY";
    out.write(magic, 6);

    const unsigned char version[2] = {1, 0};
    out.write(reinterpret_cast<const char*>(version), 2);

    const unsigned short header_len =
        static_cast<unsigned short>(header.size());
    out.write(reinterpret_cast<const char*>(&header_len), 2);
    out.write(header.data(), static_cast<std::streamsize>(header.size()));

    // Write data
    std::size_t count = 1;

    for (std::size_t dim : shape) {
        count *= dim;
    }

    out.write(
        reinterpret_cast<const char*>(data),
        static_cast<std::streamsize>(count * sizeof(float))
    );

    if (!out) {
        throw std::runtime_error(
            "Failed while writing output file: " + path
        );
    }
}

// BF16 -> FP32
std::vector<float> BF16ToFloat(const std::vector<__nv_bfloat16>& input) {
    std::vector<float> output(input.size());

    for (std::size_t i = 0; i < input.size(); ++i) {
        output[i] = __bfloat162float(input[i]);
    }

    return output;
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

// Parse command-line arguments
std::string GetPrompt(int argc, char** argv) {
    if (argc != 3 || std::string(argv[1]) != "--prompt") {
        throw std::runtime_error(
            "Usage: gpu-logits-test --prompt \"your prompt\""
        );
    }

    return argv[2];
}

int main(int argc, char** argv) {
    try {
        const std::string prompt = GetPrompt(argc, argv);

        // Model
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

        // Tokenize prompt
        std::vector<int> tokens = tokenizer.encode(prompt, true, false);
        const int prompt_len = static_cast<int>(tokens.size());
        std::cout << "Prompt length: " << prompt_len << " tokens\n";

        // Prefill
        transformer.Prefill(tokens);
        transformer.Synchronize();

        // ========================================================
        // Fetch prefill logits
        // Logits shape: [prompt_len, vocab_size]
        //
        // Only the last position is used for the first
        // autoregressive prediction.
        // ========================================================
        const std::vector<float> prefill_logits =
            transformer.Logits(prompt_len);

        const float* last_prefill_logits =
            prefill_logits.data()
            + static_cast<std::size_t>(prompt_len - 1)
            * config.vocab_size;

        SaveNpyFloat32(
            "test/dumps/cpp_prefill_logits.npy",
            last_prefill_logits,
            {
                1,
                static_cast<std::size_t>(
                    config.vocab_size
                )
            }
        );

        // First generated token
        int next_token = transformer.ArgmaxToken(prompt_len);

        // Decode
        constexpr int max_new_tokens = 100;
        constexpr int eos_token_id = 2;

        Utf8StreamDecoder stream_decoder;

        std::vector<int> generated_tokens;
        generated_tokens.reserve(max_new_tokens);

        // Generate text
        std::cout << "\nGenerated Result:\n";
        std::cout << prompt;

        for (int step = 0; step < max_new_tokens; ++step) {
            if (next_token == eos_token_id) break;

            // Decode token to text
            const std::string token_text =
                tokenizer.decode(
                    tokens.empty() ? -1 : tokens.back(), 
                    next_token
                );

            stream_decoder.print_bytes(token_text);

            generated_tokens.push_back(next_token);

            // Run one decode step
            transformer.Decode(next_token);
            transformer.Synchronize();

            // Fetch logits for this decode step
            const std::vector<float> decode_logits =
                transformer.Logits(1);

            const std::string dump_path =
                "test/dumps/cpp_decode_logits_"
                + std::to_string(step)
                + ".npy";

            SaveNpyFloat32(
                dump_path,
                decode_logits.data(),
                {
                    1,
                    static_cast<std::size_t>(config.vocab_size)
                }
            );

            next_token = transformer.ArgmaxToken(1);
        }

        std::cout << "\n";

        // Save generated token IDs
        {
            std::ofstream token_out("test/dumps/cpp_tokens.txt");

            if (!token_out.is_open()) {
                throw std::runtime_error(
                    "Failed to create test/dumps/cpp_tokens.txt"
                );
            }

            for (int token : generated_tokens) {
                token_out << token << '\n';
            }
        }

        std::cout
            << "\n"
            << "C++ inference completed.\n"
            << "Generated tokens: "
            << generated_tokens.size()
            << "\n";

        std::cout << "Logits and token IDs saved to test/dumps/\n";
    }  
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }

    return 0;
}  