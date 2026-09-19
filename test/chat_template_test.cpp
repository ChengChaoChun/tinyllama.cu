#include <iostream>
#include <string>
#include <vector>
#include <utility>
#include <stdexcept>

#include "tokenizer.h"

class ChatTemplate {
public:
    explicit ChatTemplate(Tokenizer& tokenizer)
        : tokenizer_(tokenizer)
    {}

    std::vector<int> Apply(
        const std::vector<std::pair<std::string, std::string>>& messages
    ) {
        // Build complete TinyLlama chat template
        std::string prompt;

        for (const auto& [role, content] : messages) {
            if (role != "system" &&
                role != "user" &&
                role != "assistant") {
                throw std::runtime_error(
                    "Unknown role: " + role
                );
            }

            prompt += "<|" + role + "|>\n";
            prompt += content;

            // Use a rare Unicode placeholder instead of literal "</s>".
            //
            // Runtime string: "ிждёнি♭"
            //
            // SentencePiece tokenizes this placeholder as:
            //     [30781, 28290, 30786, 30771]
            //
            // The complete sequence is replaced with the actual
            // TinyLlama EOS token ID (2) after tokenization.
            prompt += "ிждёнি♭";    
            prompt += "\n";
        }

        // add_generation_prompt=True
        prompt += "<|assistant|>\n";

        std::vector<int> tokens =
            tokenizer_.encode(
                prompt,
                false,   // no BOS
                false    // no EOS
            );

        ReplaceEosPlaceholder(tokens);

        return tokens;
    }


private:
    Tokenizer& tokenizer_;

    static constexpr int kEosTokenId = 2;

    // Unicode placeholder: "ிждёнি♭" 
    // Token IDs: 30781, 28290, 30786, 30771
    static constexpr int kPlaceholderPart1 = 30781;
    static constexpr int kPlaceholderPart2 = 28290;
    static constexpr int kPlaceholderPart3 = 30786;
    static constexpr int kPlaceholderPart4 = 30771;

    void ReplaceEosPlaceholder(std::vector<int>& tokens) {
        // ========================================================
        // Replace: [30781, 28290, 30786, 30771]
        // with: [2]
        //
        // The replacement is done in-place to avoid repeated
        // vector erase operations.
        // ========================================================
        std::size_t write = 0;

        for (std::size_t read = 0; read < tokens.size();) {
            if (read + 3 < tokens.size() &&
                tokens[read]     == kPlaceholderPart1 &&
                tokens[read + 1] == kPlaceholderPart2 &&
                tokens[read + 2] == kPlaceholderPart3 &&
                tokens[read + 3] == kPlaceholderPart4) {

                tokens[write++] = kEosTokenId;

                read += 4;
            }
            else {
                tokens[write++] = tokens[read++];
            }
        }

        tokens.resize(write);
    }
};

int main() {
    try {
        Tokenizer tokenizer(
            "models/TinyLlama-1.1B-Chat-v1.0/tokenizer.model"
        );

        // Test messages
        std::vector<std::pair<std::string, std::string>> messages = {
            {
                "system",
                "You are a helpful assistant."
            },

            {
                "user",
                "我正在學習 CUDA。我已經了解 thread、block 和 grid，但還不太理解 warp。請先解釋 warp 是什麼，然後舉一個實際的 CUDA 範例。"  
            }
        };

        // Apply Chat Template
        ChatTemplate chat_template(tokenizer);
        std::vector<int> tokens = chat_template.Apply(messages);


        // Print result
        std::cout
            << "============================================\n"
            << " TinyLlama Chat Template Test\n"
            << "============================================\n\n";

        std::cout << "Messages:\n\n";

        for (const auto& [role, content] : messages) {
            std::cout << "role    : " << role << '\n';
            std::cout << "content : " << content << "\n\n";
        }

        std::cout << "Token IDs:\n[";
        for (std::size_t i = 0; i < tokens.size(); ++i) {
            std::cout << tokens[i];

            if (i + 1 < tokens.size()) {
                std::cout << ", ";
            }
        }
        std::cout << "]\n\n";

        std::cout << "Token IDs with index:\n";
        for (std::size_t i = 0; i < tokens.size(); ++i) {
            std::cout << i << ": " << tokens[i] << '\n';
        }
        std::cout << '\n';

        std::cout << "Token Count: " << tokens.size() << '\n';
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }

    return 0;
}