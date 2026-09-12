#include "safetensor_reader.h"

#include <cerrno>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <system_error>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <nlohmann/json.hpp>

using json = nlohmann::json;

SafeTensorReader::SafeTensorReader(const std::string& filename)
    : filename_(filename)
{
    open_file();

    try {
        map_file();
        read_header();
    }
    catch (...) {
        unmap_file();

        if (fd_ != -1) {
            ::close(fd_);
            fd_ = -1;
        }

        throw;
    }
}

SafeTensorReader::~SafeTensorReader() {
    unmap_file();

    if (fd_ != -1) {
        ::close(fd_);
        fd_ = -1;
    }
}

SafeTensorReader::SafeTensorReader(
    SafeTensorReader&& other
) noexcept
    : filename_(std::move(other.filename_)),
      fd_(other.fd_),
      mapped_data_(other.mapped_data_),
      mapped_size_(other.mapped_size_),
      header_size_(other.header_size_),
      tensors_(std::move(other.tensors_))
{
    other.fd_ = -1;
    other.mapped_data_ = nullptr;
    other.mapped_size_ = 0;
    other.header_size_ = 0;
}
 
SafeTensorReader& SafeTensorReader::operator=(
    SafeTensorReader&& other ) noexcept
{
    if (this == &other) { return *this; }

    unmap_file();

    if (fd_ != -1) { ::close(fd_); }

    filename_ = std::move(other.filename_);

    fd_ = other.fd_;

    mapped_data_ = other.mapped_data_;
    mapped_size_ = other.mapped_size_;
    header_size_ = other.header_size_;

    tensors_ = std::move(other.tensors_);

    other.fd_ = -1;
    other.mapped_data_ = nullptr;
    other.mapped_size_ = 0;
    other.header_size_ = 0;

    return *this;
}

void SafeTensorReader::open_file() {
    fd_ = ::open(filename_.c_str(), O_RDONLY);

    if (fd_ == -1) {
        throw std::system_error(
            errno,
            std::generic_category(),
            "Failed to open safetensors file: " + filename_
        );
    }
}

void SafeTensorReader::map_file() {
    struct stat st{};

    if (::fstat(fd_, &st) == -1) {
        throw std::system_error(
            errno,
            std::generic_category(),
            "fstat failed: " + filename_
        );
    }

    if (st.st_size <= 0) {
        throw std::runtime_error(
            "Safetensors file is empty: " + filename_
        );
    }

    mapped_size_ = static_cast<size_t>(st.st_size);

    void* result = ::mmap(
        nullptr,
        mapped_size_,
        PROT_READ,
        MAP_PRIVATE,
        fd_,
        0
    );

    if (result == MAP_FAILED) {
        mapped_data_ = nullptr;

        throw std::system_error(
            errno,
            std::generic_category(),
            "mmap failed: " + filename_
        );
    }

    mapped_data_ = static_cast<const std::byte*>(result);
}

void SafeTensorReader::unmap_file() {
    if (mapped_data_ != nullptr) {
        ::munmap(
            const_cast<std::byte*>(mapped_data_),
            mapped_size_
        );

        mapped_data_ = nullptr;
        mapped_size_ = 0;
    }
}

// safetensors:
// [8 bytes] header_size
// [header_size bytes] JSON header
// [remaining] tensor data
void SafeTensorReader::read_header() {
    constexpr size_t header_size_field_size = sizeof(uint64_t);

    if (mapped_size_ < header_size_field_size) {
        throw std::runtime_error(
            "Invalid safetensors file: "
            "file is smaller than header size field."
        );
    }

    // safetensors 使用 little-endian uint64
    uint64_t value = 0;

    for (size_t i = 0; i < 8; ++i) {
        const auto byte = std::to_integer<uint8_t>(mapped_data_[i]);

        value |= static_cast<uint64_t>(byte) << (i * 8);
    }

    header_size_ = value;

    // 確認 header 不超過檔案
    const uint64_t header_begin = 8;

    const uint64_t data_begin = header_begin + header_size_;

    if (data_begin > mapped_size_) {
        throw std::runtime_error(
            "Invalid safetensors file: "
            "header exceeds file size."
        );
    }

    const char* header_ptr =
        reinterpret_cast<const char*>(mapped_data_ + header_begin);

    std::string json_header(
        header_ptr,
        static_cast<size_t>(header_size_)
    );

    parse_header(json_header);
}

// parse_header
void SafeTensorReader::parse_header(const std::string& json_header) {
    json j;

    try {
        j = json::parse(json_header);
    }
    catch (const json::exception& e) {
        throw std::runtime_error(
            std::string(
                "Safetensors JSON parse error: "
            ) + e.what()
        );
    }

    if (!j.is_object()) {
        throw std::runtime_error(
            "Invalid safetensors header: "
            "root must be a JSON object."
        );
    }

    tensors_.clear();

    for (const auto& [name, value] : j.items()) {
        // __metadata__ 不是 tensor
        if (name == "__metadata__") {
            continue;
        }

        if (!value.is_object()) {
            throw std::runtime_error(
                "Invalid tensor metadata: " + name
            );
        }

        // dtype
        if (!value.contains("dtype")) {
            throw std::runtime_error(
                "Tensor has no dtype: " + name
            );
        }

        const std::string dtype_string = 
            value.at("dtype").get<std::string>();

        const DType dtype = parse_dtype(dtype_string);

        // ----------------------------------------------------
        // shape
        //
        // 這裡不保存 shape 到 TensorFileInfo。
        //
        // 但我們仍然解析它，確認格式正確。
        // ModelLoader / ModelConfig 負責 runtime shape。
        // ----------------------------------------------------
        if (!value.contains("shape")) {
            throw std::runtime_error(
                "Tensor has no shape: " + name
            );
        }

        if (!value.at("shape").is_array()) {
            throw std::runtime_error(
                "Tensor shape is not an array: " + name
            );
        }

        // data_offsets
        if (!value.contains("data_offsets")) {
            throw std::runtime_error(
                "Tensor has no data_offsets: " + name
            );
        }

        const auto& offsets = value.at("data_offsets");

        if (!offsets.is_array() || offsets.size() != 2) {
            throw std::runtime_error("Invalid data_offsets: " + name);
        }

        TensorFileInfo info;
        info.dtype = dtype;
        info.offset_begin = offsets[0].get<uint64_t>();
        info.offset_end = offsets[1].get<uint64_t>();


        // 驗證 offset
        validate_tensor(info);

        tensors_.emplace(name, info);
    }
}

// validate_tensor
// offset 是相對於 tensor data 區域的 offset。
// tensor data 開始位置： 8 + header_size
void SafeTensorReader::validate_tensor(const TensorFileInfo& info) const
{
    if (info.offset_end < info.offset_begin) {
        throw std::runtime_error(
            "Invalid tensor offsets: "
            "offset_end < offset_begin."
        );
    }

    const uint64_t data_begin = sizeof(uint64_t) + header_size_;
    const uint64_t tensor_begin = data_begin + info.offset_begin;
    const uint64_t tensor_end = data_begin + info.offset_end;

    if (tensor_begin > mapped_size_ || tensor_end > mapped_size_) {
        throw std::runtime_error("Tensor data exceeds safetensors file.");
    }
}

const TensorFileInfo& SafeTensorReader::tensor_info(
    const std::string& tensor_name) const 
{
    const auto it = tensors_.find(tensor_name);

    if (it == tensors_.end()) {
        throw std::runtime_error("Tensor not found: " + tensor_name);
    }

    return it->second;
}

// view 不做 copy，直接回傳 mmap memory 的 view。
// IMPORTANT:
// TensorView 不能活得比 SafeTensorReader 久。
TensorView SafeTensorReader::view(const std::string& tensor_name) const {
    const TensorFileInfo& info = tensor_info(tensor_name);

    const uint64_t data_begin = sizeof(uint64_t) + header_size_;
    const uint64_t tensor_begin = data_begin + info.offset_begin;
    const uint64_t tensor_size = info.offset_end - info.offset_begin;

    const std::byte* ptr = mapped_data_ + tensor_begin;

    return TensorView {
        info.dtype,
        std::span<const std::byte>(
            ptr,
            static_cast<size_t>(tensor_size)
        )
    };
}

DType SafeTensorReader::parse_dtype(const std::string& dtype) {
    if (dtype == "BF16") { return DType::BF16; }
    if (dtype == "F32") { return DType::FP32; }
    if (dtype == "F16") { return DType::FP16; }

    throw std::runtime_error(
        "Unsupported safetensors dtype: " + dtype
    );
}

const char* SafeTensorReader::dtype_to_string(DType dtype) noexcept {
    switch (dtype) {
        case DType::BF16:
            return "BF16";

        case DType::FP32:
            return "F32";

        case DType::FP16:
            return "F16";
    }

    return "UNKNOWN";
}


// ============================================================
// print_tensors
// ============================================================

void SafeTensorReader::print_tensors() const {
    std::cout << "\nTensor count: " << tensors_.size() << "\n\n";

    for (const auto& [name, info] : tensors_) {
        std::cout << "--------------------------------\n";

        std::cout << "Name:\n" << name << "\n";

        std::cout << "dtype: " << dtype_to_string(info.dtype) << "\n";

        std::cout
            << "offset: " << info.offset_begin
            << " ~ " << info.offset_end << "\n";

        std::cout << "size: " << info.size_bytes() << " bytes\n";
    }
}