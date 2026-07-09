#ifndef CUSGA_KERNELUTILS_CUH
#define CUSGA_KERNELUTILS_CUH
#include <format>
#include <source_location>
#include <stdexcept>

namespace cuSGA::KernelUtils {
    inline constexpr ::uint8_t WARP_SIZE{32};
    inline constexpr ::uint8_t MAX_WARPS_PER_BLOCK{32};

    inline void checkLastCudaError(const std::source_location location = std::source_location::current()) {
        // Get last CUDA error and throw exception if error occurred
        if (const auto CUDAError{::cudaGetLastError()}; CUDAError != ::cudaSuccess) {
            throw std::runtime_error{std::format("CUDA ERROR [{} -> {}:{}]: {}", location.file_name(), location.function_name(), location.line(), cudaGetErrorString(CUDAError))};
        }
    }

    template <auto Kernel, typename... Args>
    void launchKernel(const ::size_t dynamicSMemSize, const int maxBlockSize, const ::size_t numElements, const bool sync, Args&&... args) {
        // Compute optimal block size if necessary
        static auto minGridSize{0}, blockSize{0};
        if (blockSize == 0) {
            ::cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, Kernel, dynamicSMemSize, maxBlockSize);
            checkLastCudaError();
        }

        // Size grid
        const auto gridSize{(numElements + blockSize - 1) / blockSize};

        // Launch kernel
        Kernel<<<gridSize, blockSize, dynamicSMemSize>>>(std::forward<Args>(args)...);
        checkLastCudaError();

        // Synchronize with device if necessary
        if (sync) {
            ::cudaDeviceSynchronize();
            checkLastCudaError();
        }
    }
} // cuSGA::KernelUtils

#endif //CUSGA_KERNELUTILS_CUH
