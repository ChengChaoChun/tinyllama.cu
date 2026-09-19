#include <chrono>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "model_config.h"
#include "tokenizer.h"
#include "chat_template.h"

#include "gpu_model_loader.h"
#include "gpu_transformer.h"


// Benchmark Configuration
constexpr int kMaxNewTokens = 200;

// Warmup 次數
// 用來讓 CUDA / cuBLAS 等初始化完成，避免第一次執行影響結果。
constexpr int kWarmupRuns = 5;

// 正式 benchmark 次數
constexpr int kBenchmarkRuns = 10;

constexpr int kEosTokenId = 2;


class CpuTimer {
public:
    void Start() { start_ = Clock::now(); }

    double StopMilliseconds() const {
        const auto end = Clock::now();

        return std::chrono::duration<double, std::milli>(
            end - start_
        ).count();
    }

private:
    using Clock = std::chrono::steady_clock;

    Clock::time_point start_;
};

int RunGeneration(
    GPUTransformer& transformer,
    const std::vector<int>& tokens,
    int max_new_tokens
) {
    if (tokens.empty()) {
        throw std::runtime_error(
            "Cannot run generation with empty prompt."
        );
    }

    const int prompt_len = static_cast<int>(tokens.size());

    // Reset previous inference state
    transformer.Reset();

    // prefill 
    transformer.Prefill(tokens);
    int next_token = transformer.ArgmaxToken(prompt_len);

    // Autoregressive Decode
    int generated_tokens = 0;

    for (int step = 0; step < max_new_tokens; ++step) {
        if (next_token == kEosTokenId) {
            break;
        }

        // Feed generated token back into model.
        transformer.Decode(next_token);

        // GPU Argmax + 4-byte D→H
        next_token = transformer.ArgmaxToken(1);

        ++generated_tokens;
    }

    return generated_tokens;
}

int main() {
    try {
        ModelConfig config(
            "models/TinyLlama-1.1B-Chat-v1.0/config.json"
        );

        GPUModelLoader gpu_loader(
            "models/TinyLlama-1.1B-Chat-v1.0/model.safetensors",
            config
        );

        GPUModelWeights gpu_weights = gpu_loader.Load();

        Tokenizer tokenizer(
            "models/TinyLlama-1.1B-Chat-v1.0/tokenizer.model"
        );
        ChatTemplate chat_template(tokenizer);

        GPUTransformer transformer(config, gpu_weights);

        // Tokenize
        const std::string prompt = 
        "Imagine you are an engineer living on a Mars colony in the year 2147. "
        "One night, you receive a transmission from Earth with a timestamp indicating "
        "that it was sent 137 years ago. The message contains only one sentence: "
        "\"Do not trust the ship arriving tomorrow.\" "
        "Explain what this message could mean, then turn your reasoning into a science-fiction story. "
        "The story should contain at least three unexpected twists, but every twist must be "
        "logically consistent with information revealed earlier. "
        "End the story with a surprising but believable explanation of who sent the original "
        "message and why.";

        std::vector<std::pair<std::string, std::string>> messages = {
            { "system", "You are a helpful assistant." },
            { "user", prompt }  
        };

        std::vector<int> tokens = chat_template.Apply(messages);

        if (tokens.empty()) {
            throw std::runtime_error(
                "Benchmark prompt produced no tokens."
            );
        }

        const int prompt_len = static_cast<int>(tokens.size());

        std::cout 
            << "\nBenchmark Prompt\n"
            << "--------------------------------------------\n";

        for (const auto& [role, content] : messages) {
            std::cout << "[" << role << "]\n";
            std::cout << content << "\n\n";
        }

        std::cout
            << "--------------------------------------------\n\n";

        std::cout
            << "============================================\n"
            << " TinyLlama CUDA Performance Benchmark\n"
            << "============================================\n\n";

        std::cout
            << "Prompt length      : " << prompt_len << " tokens\n";

        std::cout
            << "Max new tokens     : " << kMaxNewTokens << "\n";

        std::cout
            << "Warmup runs        : " << kWarmupRuns << "\n";

        std::cout
            << "Benchmark runs     : " << kBenchmarkRuns << "\n\n";

        // Warmup
        std::cout << "Running warmup...\n";

        for (int run = 0; run < kWarmupRuns; ++run) {
            RunGeneration(transformer, tokens, kMaxNewTokens);
        }

        std::cout << "Warmup complete.\n\n";

        // 1. Prefill Latency
        std::cout << "Benchmarking Prefill...\n";

        double total_prefill_ms = 0.0;

        for (int run = 0; run < kBenchmarkRuns; ++run) {
            transformer.Reset();

            CpuTimer timer;
            timer.Start();

            transformer.Prefill(tokens);

            transformer.Synchronize();

            const double elapsed_ms = timer.StopMilliseconds();

            total_prefill_ms += elapsed_ms;
        }

        const double avg_prefill_ms =
            total_prefill_ms / static_cast<double>(kBenchmarkRuns);

        // 2. Decode Latency / Token
        std::cout << "Benchmarking Decode...\n";

        double total_decode_ms = 0.0;

        int total_generated_tokens = 0;

        for (int run = 0; run < kBenchmarkRuns; ++run) {
            transformer.Reset();

            transformer.Prefill(tokens);
            int next_token = transformer.ArgmaxToken(prompt_len);

            // Start Decode timer
            CpuTimer timer;
            timer.Start();

            int generated_this_run = 0;

            for (int step = 0; step < kMaxNewTokens; ++step) {
                if (next_token == kEosTokenId) {
                    break;
                }

                transformer.Decode(next_token);
                next_token = transformer.ArgmaxToken(1);

                ++generated_this_run;
            }


            // Stop Decode timer
            const double elapsed_ms = timer.StopMilliseconds();

            total_decode_ms += elapsed_ms;

            total_generated_tokens += generated_this_run;
        }

        // Calculate Decode Results
        const double avg_decode_ms =
            total_decode_ms / static_cast<double>(kBenchmarkRuns);

        const double avg_generated_tokens =
            static_cast<double>(total_generated_tokens) /
            static_cast<double>(kBenchmarkRuns);

        if (avg_generated_tokens <= 0.0) {
            throw std::runtime_error(
                "No tokens were generated during benchmark."
            );
        }

        // Average end-to-end latency for one generated token.
        const double decode_ms_per_token = 
            avg_decode_ms / avg_generated_tokens;

        // tokens / second
        const double token_throughput = 1000.0 / decode_ms_per_token;

        // Print Results
        std::cout
            << "\n"
            << "============================================\n"
            << " Benchmark Results\n"
            << "============================================\n\n";

        std::cout << std::fixed << std::setprecision(3);

        std::cout
            << "Prompt Length      : "
            << prompt_len
            << " tokens\n";

        std::cout
            << "Generated Tokens   : "
            << avg_generated_tokens
            << " tokens/run\n";

        std::cout
            << "Prefill Latency    : "
            << avg_prefill_ms
            << " ms\n";

        std::cout
            << "Decode Latency     : "
            << decode_ms_per_token
            << " ms/token\n";

        std::cout
            << "Token Throughput   : " 
            << token_throughput 
            << " tokens/s\n";


        std::cout << "\n============================================\n";

        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "\nError: " << e.what() << '\n';
        return 1;
    }
}