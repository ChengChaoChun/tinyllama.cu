#pragma once

#include <cuda_runtime.h>
#include <utility>
#include <stdexcept>
#include <string>

#ifndef CHECK_CUDA
#define CHECK_CUDA(call)                                                     \
    do {                                                                     \
        cudaError_t err = (call);                                            \
        if (err != cudaSuccess) {                                            \
            throw std::runtime_error(                                        \
                std::string("CUDA Error: ") + cudaGetErrorString(err));       \
        }                                                                    \
    } while (0)
#endif

template<typename T>
class DeviceBuffer {
public:
    DeviceBuffer() = default;

    explicit DeviceBuffer(size_t count) {
        allocate(count);
    }

    ~DeviceBuffer() {
        release();
    }
    
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    DeviceBuffer(DeviceBuffer&& other) noexcept {
        move_from(std::move(other));
    }
    
    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            release();
            move_from(std::move(other));
        }
        return *this;
    }

public:
    void allocate(size_t count) {
        release();

        if (count == 0) return;

        CHECK_CUDA(cudaMalloc(&ptr_, sizeof(T) * count));
        size_ = count;
    }

    void release() noexcept {
        if (ptr_) {
            cudaFree(ptr_);
            ptr_ = nullptr;
        }
        size_ = 0;
    }

    void memset(int value = 0) {
        if (ptr_) {
            CHECK_CUDA(cudaMemset(ptr_, value, sizeof(T) * size_));
        }
    }

    void copy_from_host(const T* host_ptr, size_t count) {
        if (count > size_)
            throw std::runtime_error("DeviceBuffer overflow");

        CHECK_CUDA(cudaMemcpy(
            ptr_,
            host_ptr,
            sizeof(T) * count,
            cudaMemcpyHostToDevice));
    }

    void copy_to_host(T* host_ptr, size_t count) const {
        if (count > size_)
            throw std::runtime_error("DeviceBuffer overflow");

        CHECK_CUDA(cudaMemcpy(
            host_ptr,
            ptr_,
            sizeof(T) * count,
            cudaMemcpyDeviceToHost));
    }

public:
    T* data() noexcept { return ptr_; }
    const T* data() const noexcept { return ptr_; }

    size_t size() const noexcept { return size_; }

    size_t bytes() const noexcept { return sizeof(T) * size_; }

    explicit operator bool() const noexcept { return ptr_ != nullptr; }

private:
    void move_from(DeviceBuffer&& other) noexcept {
        ptr_  = std::exchange(other.ptr_, nullptr);
        size_ = std::exchange(other.size_, 0);
    }

private:
    T* ptr_ = nullptr;
    size_t size_ = 0;
};