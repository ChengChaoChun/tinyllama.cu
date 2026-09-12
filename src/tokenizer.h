#pragma once

#include <string>
#include <vector>
#include <string_view>
#include <memory>

// Forward declaration 避免在 .hpp 引入外部標頭檔
namespace sentencepiece {
    class SentencePieceProcessor;
}

class Tokenizer {
public:
    // 傳入 SentencePiece 模型路徑 (例如 tokenizer.model)
    explicit Tokenizer(const std::string& model_path);
    ~Tokenizer();  

    // 禁用 Copy，允許 Move
    Tokenizer(const Tokenizer&) = delete;
    Tokenizer& operator=(const Tokenizer&) = delete;
    Tokenizer(Tokenizer&&) noexcept;
    Tokenizer& operator=(Tokenizer&&) noexcept;

    // 核心介面：文字 轉 Tokens
    [[nodiscard]] std::vector<int> encode(std::string_view text, bool bos = true, bool eos = false) const;

    // 核心介面：Token 轉 文字 (SentencePiece 會自動處理空格與 byte fallback)
    // decode 可以要有一個參數的，因為自回歸生成不需要使用 vector
    [[nodiscard]] std::string decode(int prev_token, int token) const;
    [[nodiscard]] std::string decode(const std::vector<int>& tokens) const;

    // 安全列印工具
    static void safe_print(std::string_view piece);

    [[nodiscard]] int vocab_size() const noexcept;
    [[nodiscard]] int bos_id() const noexcept;
    [[nodiscard]] int eos_id() const noexcept;

private:
    // 使用 std::unique_ptr 搭配 PImpl 模式隱藏實作
    std::unique_ptr<sentencepiece::SentencePieceProcessor> spp_;
};

