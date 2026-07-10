#ifndef CUSGA_KERNELUTILS_CUH
#define CUSGA_KERNELUTILS_CUH
#include <format>
#include <source_location>
#include <stdexcept>

namespace cuSGA::KernelUtils {
    inline constexpr ::uint8_t WARP_SIZE{32};
    inline constexpr ::uint8_t MAX_WARPS_PER_BLOCK{32};

    // cudaGetLastError() wrapper
    inline void cudaCheckLastError(const ::std::source_location location = ::std::source_location::current()) {
        // Get last CUDA error and throw exception if error occurred
        if (const auto cudaError{::cudaGetLastError()}; cudaError != ::cudaSuccess) {
            throw ::std::runtime_error{::std::format("CUDA ERROR [{} -> {}:{}]: {}", location.file_name(), location.function_name(), location.line(), ::cudaGetErrorString(cudaError))};
        }
    }

    // cudaMalloc() wrapper
    template <typename T>
    void cudaMalloc(T** const d_ptr, const ::size_t size) {
        // Forward call
        ::cudaMalloc(d_ptr, size);

        // Check for errors
        cudaCheckLastError();
    }

    // cudaFree() wrapper
    inline void cudaFree(void* const d_ptr) {
        // Forward call
        ::cudaFree(d_ptr);

        // Check for errors
        cudaCheckLastError();
    }

    // cudaMemcpy() wrapper
    inline void cudaMemcpy(void* const dst, const void* const src, const ::size_t count, const ::cudaMemcpyKind kind) {
        // Forward call
        ::cudaMemcpy(dst, src, count, kind);

        // Check for errors
        cudaCheckLastError();
    }

    // cudaMemset() wrapper
    inline void cudaMemset(void* const d_ptr, const int value, const ::size_t count) {
        // Forward call
        ::cudaMemset(d_ptr, value, count);

        // Check for errors
        cudaCheckLastError();
    }

    template <auto Kernel, typename... Args>
    void cudaLaunchKernel(const ::size_t dynamicSMemSize, const int maxBlockSize, const ::size_t numElements, const bool sync, Args&&... args) {
        // Compute optimal block size if necessary
        static auto minGridSize{0}, blockSize{0};
        if (blockSize == 0) {
            ::cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, Kernel, dynamicSMemSize, maxBlockSize);
            cudaCheckLastError();
        }

        // Size grid
        const auto gridSize{(numElements + blockSize - 1) / blockSize};

        // Launch kernel
        Kernel<<<gridSize, blockSize, dynamicSMemSize>>>(::std::forward<Args>(args)...);
        cudaCheckLastError();

        // Synchronize with device if necessary
        if (sync) {
            ::cudaDeviceSynchronize();
            cudaCheckLastError();
        }
    }
} // cuSGA::KernelUtils

#endif //CUSGA_KERNELUTILS_CUH
