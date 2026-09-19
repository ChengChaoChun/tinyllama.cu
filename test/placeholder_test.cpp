#include <iostream>
#include <string>
#include <vector>

#include "tokenizer.h"

void PrintTokens(
    const std::string& input,
    const std::vector<int>& tokens
) {
    std::cout << "Input: " << input << "\n";
    std::cout << "Tokens: [";

    for (std::size_t i = 0; i < tokens.size(); ++i) {
        std::cout << tokens[i];

        if (i + 1 < tokens.size()) {
            std::cout << ", ";
        }
    }

    std::cout << "]\n\n";
}

std::vector<int> Encode(
    Tokenizer& tokenizer,
    const std::string& input
) {
    return tokenizer.encode(
        input,
        false,
        false
    );
}

int main() {
    Tokenizer tokenizer(
        "models/TinyLlama-1.1B-Chat-v1.0/tokenizer.model"
    );

    // Rare Unicode sentinel.
    //
    // Runtime string:
    //   ிждёнি♭ 
    //
    // The characters are intentionally unusual so that normal
    // user input is very unlikely to contain this sequence.
    constexpr const char* kPlaceholder = "ிждёнি♭";  

    std::cout << "============================================\n";
    std::cout << " SentencePiece Unicode Placeholder Test\n";
    std::cout << "============================================\n\n";

    std::cout << "Placeholder: " << kPlaceholder << "\n\n";

    // ------------------------------------------------------------
    // Test 1: Placeholder by itself
    // ------------------------------------------------------------
    {
        const std::string input = kPlaceholder;

        auto tokens = Encode(tokenizer, input);

        std::cout << "[Test 1] Placeholder alone\n";
        PrintTokens(input, tokens);
    }

    // ------------------------------------------------------------
    // Test 2: English text + placeholder
    // ------------------------------------------------------------
    {
        const std::string input =
            std::string("hello") + kPlaceholder;

        auto tokens = Encode(tokenizer, input);

        std::cout << "[Test 2] English + placeholder\n";
        PrintTokens(input, tokens);
    }

    // ------------------------------------------------------------
    // Test 3: Placeholder + English text
    // ------------------------------------------------------------
    {
        const std::string input =
            std::string(kPlaceholder) + "world";

        auto tokens = Encode(tokenizer, input);

        std::cout << "[Test 3] Placeholder + English\n";
        PrintTokens(input, tokens);
    }

    // ------------------------------------------------------------
    // Test 4: English text + placeholder + English text
    // ------------------------------------------------------------
    {
        const std::string input =
            std::string("hello") +
            kPlaceholder +
            "world";

        auto tokens = Encode(tokenizer, input);

        std::cout << "[Test 4] English + placeholder + English\n";
        PrintTokens(input, tokens);
    }

    // ------------------------------------------------------------
    // Test 5: English sentence ending with '.'
    //
    // This is especially important because the previous placeholder
    // could merge with punctuation from the preceding text.
    // ------------------------------------------------------------
    {
        const std::string input =
            std::string("NVIDIA.") + kPlaceholder;

        auto tokens = Encode(tokenizer, input);

        std::cout << "[Test 5] Text ending with '.' + placeholder\n";
        PrintTokens(input, tokens);
    }

    // ------------------------------------------------------------
    // Test 6: English sentence ending with '?'
    //
    // This checks the exact boundary that caused the previous
    // ASCII placeholder to merge with '?'.
    // ------------------------------------------------------------
    {
        const std::string input =
            std::string("How are you?") + kPlaceholder;

        auto tokens = Encode(tokenizer, input);

        std::cout << "[Test 6] Text ending with '?' + placeholder\n";
        PrintTokens(input, tokens);
    }

    // ------------------------------------------------------------
    // Test 7: English sentence ending with '!'
    // ------------------------------------------------------------
    {
        const std::string input =
            std::string("Hello!") + kPlaceholder;

        auto tokens = Encode(tokenizer, input);

        std::cout << "[Test 7] Text ending with '!' + placeholder\n";
        PrintTokens(input, tokens);
    }

    // ------------------------------------------------------------
    // Test 8: Chinese + placeholder
    // ------------------------------------------------------------
    {
        const std::string input =
            std::string("你好，這是一個測試。") +
            kPlaceholder;

        auto tokens = Encode(tokenizer, input);

        std::cout << "[Test 8] Chinese + placeholder\n";
        PrintTokens(input, tokens);
    }

    // ------------------------------------------------------------
    // Test 9: Chinese text + placeholder + English
    // ------------------------------------------------------------
    {
        const std::string input =
            std::string("你好，這是一個測試。") +
            kPlaceholder +
            "hello";

        auto tokens = Encode(tokenizer, input);

        std::cout << "[Test 9] Chinese + placeholder + English\n";
        PrintTokens(input, tokens);
    }

    // ------------------------------------------------------------
    // Test 10: Placeholder after a newline
    // ------------------------------------------------------------
    {
        const std::string input =
            std::string("NVIDIA develops CUDA.\n") +
            kPlaceholder;

        auto tokens = Encode(tokenizer, input);

        std::cout << "[Test 10] Newline + placeholder\n";
        PrintTokens(input, tokens);
    }

    // ------------------------------------------------------------
    // Test 11: Placeholder before a newline
    // ------------------------------------------------------------
    {
        const std::string input =
            std::string("NVIDIA develops CUDA.") +
            kPlaceholder +
            "\n";

        auto tokens = Encode(tokenizer, input);

        std::cout << "[Test 11] Placeholder + newline\n";
        PrintTokens(input, tokens);
    }

    // ------------------------------------------------------------
    // Test 12: Placeholder surrounded by spaces
    // ------------------------------------------------------------
    {
        const std::string input =
            std::string("hello ") +
            kPlaceholder +
            " world";

        auto tokens = Encode(tokenizer, input);

        std::cout << "[Test 12] Placeholder surrounded by spaces\n";
        PrintTokens(input, tokens);
    }

    // ------------------------------------------------------------
    // Test 13: Placeholder with spaces on both sides
    // ------------------------------------------------------------
    {
        const std::string input =
            std::string("hello ") +
            kPlaceholder +
            " world";

        auto tokens = Encode(tokenizer, input);

        std::cout << "[Test 13] Placeholder with surrounding spaces\n";
        PrintTokens(input, tokens);
    }

    // ------------------------------------------------------------
    // Test 14: Placeholder followed by punctuation
    // ------------------------------------------------------------
    {
        const std::string input =
            std::string("hello") +
            kPlaceholder +
            ".";

        auto tokens = Encode(tokenizer, input);

        std::cout << "[Test 14] Placeholder + punctuation\n";
        PrintTokens(input, tokens);
    }

    // ------------------------------------------------------------
    // Test 15: Placeholder between two sentences
    // ------------------------------------------------------------
    {
        const std::string input =
            std::string("Hello.") +
            kPlaceholder +
            "How are you?";

        auto tokens = Encode(tokenizer, input);

        std::cout << "[Test 15] Placeholder between sentences\n";
        PrintTokens(input, tokens);
    }

    // ------------------------------------------------------------
    // Test 16: Multiple placeholders
    //
    // This checks whether the sentinel remains stable when it appears
    // more than once in the same input.
    // ------------------------------------------------------------
    {
        const std::string input =
            std::string("hello") +
            kPlaceholder +
            "world" +
            kPlaceholder +
            "test";

        auto tokens = Encode(tokenizer, input);

        std::cout << "[Test 16] Multiple placeholders\n";
        PrintTokens(input, tokens);
    }

    return 0;
}