#include "chat_template.h"

#include <stdexcept>

ChatTemplate::ChatTemplate(Tokenizer& tokenizer)
    : tokenizer_(tokenizer)
{}

std::vector<int> ChatTemplate::Apply(
    const std::vector<std::pair<std::string, std::string>>& messages
) {
    std::string prompt;

    // Build complete TinyLlama chat template.
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

        // TinyLlama uses </s> after each message.
        //
        // Literal "</s>" cannot be directly inserted because
        // SentencePiece tokenizes it as ordinary text.
        //
        // Use a rare Unicode placeholder and replace its
        // token sequence with the EOS token ID after tokenization.
        prompt += "ிждёнি♭"; 
        prompt += "\n";
    }

    // add_generation_prompt=True
    prompt += "<|assistant|>\n";

    std::vector<int> tokens =
        tokenizer_.encode(
            prompt,
            false,  // no BOS
            false   // no EOS
        );

    ReplaceEosPlaceholder(tokens);

    return tokens;
}

void ChatTemplate::ReplaceEosPlaceholder(
    std::vector<int>& tokens
) {
    // Replace: 30781, 28290, 30786, 30771] with: [2]
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