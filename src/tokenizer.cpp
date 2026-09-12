#include "tokenizer.h"
#include <sentencepiece_processor.h>
#include <stdexcept>
#include <iostream>
#include <cctype>

Tokenizer::Tokenizer(const std::string& model_path)
    : spp_(std::make_unique<sentencepiece::SentencePieceProcessor>()) {
    
    const auto status = spp_->Load(model_path);
    if (!status.ok()) {
        throw std::runtime_error("Failed to load SentencePiece model: " + status.ToString());
    }
}

Tokenizer::~Tokenizer() = default;
Tokenizer::Tokenizer(Tokenizer&&) noexcept = default;
Tokenizer& Tokenizer::operator=(Tokenizer&&) noexcept = default;

std::vector<int> Tokenizer::encode(std::string_view text, bool bos, bool eos) const {
    std::vector<int> tokens;
    
    // 呼叫 SentencePiece 原生 Encode 介面
    const auto status = spp_->Encode(std::string(text), &tokens);
    if (!status.ok()) {
        throw std::runtime_error("Encoding failed: " + status.ToString());
    }

    // 手動處理 BOS / EOS 標籤
    if (bos) {
        tokens.insert(tokens.begin(), spp_->bos_id());
    }
    if (eos) {
        tokens.push_back(spp_->eos_id());
    }

    return tokens;
}

/*
std::string Tokenizer::decode(int prev_token, int token) const {
    std::string piece = spp_->IdToPiece(token);

    // 如果緊接在 BOS 句首後面
    if (prev_token == spp_->bos_id()) {
        // 直接用 " " 比對，並用 .size() 裁切，完全不用記 3 個 bytes 或十六進位！
        if (std::string piece = spp_->IdToPiece(token); piece.starts_with(" ")) {
            static constexpr std::string_view Space = " ";
            return std::string(piece.substr(Space.size()));  
        }
    }

    std::string text;
    spp_->Decode(std::vector<int>{token}, &text);
    return text;  
}
*/
std::string Tokenizer::decode(int prev_token, int token) const
{
    std::string piece = spp_->IdToPiece(token);

    // ------------------------------------------------------------
    // 1. Byte token
    //    例如 <0xE4> → raw byte 0xE4
    // ------------------------------------------------------------
    if (piece.size() == 6 &&
        piece.starts_with("<0x") &&
        piece.back() == '>')
    {
        int value = std::stoi(piece.substr(3, 2), nullptr, 16);

        return std::string(1, static_cast<char>(value));
    }

    // ------------------------------------------------------------
    // 2. SentencePiece 的 ▁ 表示「前面有一個空格」
    //
    //    ▁Hello → " Hello"
    //    ▁您好  → " 您好"
    // ------------------------------------------------------------
    constexpr std::string_view space_marker = "▁";

    if (piece.starts_with(space_marker)) {
        piece.replace(0, space_marker.size(), " ");
    }

    // ------------------------------------------------------------
    // 3. 如果是 BOS 後面的第一個 token，
    //    SentencePiece 原本的 ▁ 不應該變成真正的空格
    // ------------------------------------------------------------
    if (prev_token == spp_->bos_id() && piece.starts_with(" ")) {
        piece.erase(0, 1);
    }

    return piece;
}    

std::string Tokenizer::decode(const std::vector<int>& tokens) const {
    std::string text;
    spp_->Decode(tokens, &text);
    return text;
}

void Tokenizer::safe_print(std::string_view piece) {
    if (piece.empty()) return;
    if (piece.size() == 1) {
        unsigned char ch = piece[0];
        if (!std::isprint(ch) && !std::isspace(ch)) {
            return; // 濾除不可列印字元
        }
    }
    std::cout << piece << std::flush;
}

int Tokenizer::vocab_size() const noexcept { return spp_->GetPieceSize(); }
int Tokenizer::bos_id() const noexcept { return spp_->bos_id(); }
int Tokenizer::eos_id() const noexcept { return spp_->eos_id(); }