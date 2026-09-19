#pragma once

#include <string>
#include <utility>
#include <vector>

#include "tokenizer.h"

class ChatTemplate {
public:
    explicit ChatTemplate(Tokenizer& tokenizer);

    std::vector<int> Apply(
        const std::vector<std::pair<std::string, std::string>>& messages
    );  

private:
    Tokenizer& tokenizer_;

    static constexpr int kEosTokenId = 2;

    // Unicode placeholder: "ிждёнি♭"
    //
    // SentencePiece token IDs:
    // [30781, 28290, 30786, 30771]
    static constexpr int kPlaceholderPart1 = 30781;
    static constexpr int kPlaceholderPart2 = 28290;
    static constexpr int kPlaceholderPart3 = 30786;
    static constexpr int kPlaceholderPart4 = 30771;

    void ReplaceEosPlaceholder(std::vector<int>& tokens);
};