#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

enum class DType {
    BF16,
    FP32,
    FP16
};

struct TensorFileInfo {
    DType dtype;

    uint64_t offset_begin;
    uint64_t offset_end;

    uint64_t size_bytes() const noexcept {
        return offset_end - offset_begin;
    }
};

struct TensorView {
    DType dtype;
    std::span<const std::byte> bytes;
};


// ============================================================
// SafeTensorReader
//
// 負責：
// 1. 開啟 safetensors
// 2. mmap 整個檔案
// 3. 讀取 header size
// 4. parse JSON header
// 5. 建立 TensorFileInfo
// 6. 根據 tensor name 回傳 TensorView
//
// 不負責：
// - BF16 -> FP32
// - CUDA
// - GPU memory
// - ModelWeights
// - Transformer
// ============================================================
class SafeTensorReader {
public:
    explicit SafeTensorReader(const std::string& filename);
    ~SafeTensorReader();  

    SafeTensorReader(const SafeTensorReader&) = delete;
    SafeTensorReader& operator=(const SafeTensorReader&) = delete;

    SafeTensorReader(SafeTensorReader&& other) noexcept;
    SafeTensorReader& operator=(SafeTensorReader&& other) noexcept;

    // 取得 tensor view，回傳的 TensorView 不擁有資料，reader 必須活著。
    TensorView view(const std::string& tensor_name) const;

    const TensorFileInfo& tensor_info(
        const std::string& tensor_name
    ) const;

    const std::unordered_map<std::string, TensorFileInfo>&
    tensors() const noexcept {
        return tensors_;
    }

    // Debug
    void print_tensors() const;

    // 基本資訊
    uint64_t header_size() const noexcept { return header_size_; }

    size_t mapped_size() const noexcept { return mapped_size_; }

private:
    // File
    std::string filename_;
    int fd_{-1};

    // mmap
    const std::byte* mapped_data_{nullptr};
    size_t mapped_size_{0};

    // safetensors header
    uint64_t header_size_{0};

    // tensor metadata
    // key: tensor name
    // value:dtype + offsets
    std::unordered_map<std::string, TensorFileInfo> tensors_;

private:
    void open_file();
    void map_file();
    void unmap_file();

    void read_header();
    void parse_header(const std::string& json_header);

    void validate_tensor(const TensorFileInfo& info) const;

    static DType parse_dtype(const std::string& dtype);

    static const char* dtype_to_string(DType dtype) noexcept;
};