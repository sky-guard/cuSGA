#ifndef CUSGA_KERNELUTILS_CUH
#define CUSGA_KERNELUTILS_CUH
#include <format>
#include <source_location>
#include <stdexcept>

namespace cuSGA::KernelUtils {
    // Kernel Utils related constants
    inline constexpr ::size_t WARP_SIZE{32};
    inline constexpr ::size_t MAX_WARPS_PER_BLOCK{32};

    // cudaGetLastError() wrapper
    __forceinline__ inline void cudaCheckLastError(const ::std::source_location location = ::std::source_location::current()) {
        // Get last CUDA error and throw exception if error occurred
        if (const auto cudaError{::cudaGetLastError()}; cudaError != ::cudaSuccess) {
            throw ::std::runtime_error{::std::format("CUDA ERROR [{} -> {}:{}]: {}", location.file_name(), location.function_name(), location.line(), ::cudaGetErrorString(cudaError))};
        }
    }

    // cudaDeviceSynchronize() wrapper
    __forceinline__ inline void cudaDeviceSynchronize() {
        // Forward call
        ::cudaDeviceSynchronize();

        // Check for errors
        cudaCheckLastError();
    }

    // cudaMalloc() wrapper
    template <typename T>
    __forceinline__ void cudaMalloc(T** const d_ptr, const ::size_t size) {
        // Forward call
        ::cudaMalloc(d_ptr, size);

        // Check for errors
        cudaCheckLastError();
    }

    template <typename T>
    __forceinline__ void cudaMallocAsync(T** const d_ptr, const ::size_t size, const cudaStream_t& stream = cudaStreamDefault) {
        // Forward call
        ::cudaMallocAsync(d_ptr, size, stream);

        // Check for errors
        cudaCheckLastError();
    }

    // cudaMallocHost() wrapper
    template <typename T>
    __forceinline__ void cudaMallocHost(T** const d_ptr, const ::size_t size) {
        // Forward call
        ::cudaMallocHost(d_ptr, size);

        // Check for errors
        cudaCheckLastError();
    }

    // cudaFree() wrapper
    __forceinline__ inline void cudaFree(void* const d_ptr) {
        // Forward call
        ::cudaFree(d_ptr);

        // Check for errors
        cudaCheckLastError();
    }

    // cudaFreeAsync() wrapper
    __forceinline__ inline void cudaFreeAsync(void* const d_ptr, const cudaStream_t& stream = cudaStreamDefault) {
        // Forward call
        ::cudaFreeAsync(d_ptr, stream);

        // Check for errors
        cudaCheckLastError();
    }

    // cudaFreeHost() wrapper
    __forceinline__ inline void cudaFreeHost(void* const pinned_ptr) {
        // Forward call
        ::cudaFreeHost(pinned_ptr);

        // Check for errors
        cudaCheckLastError();
    }

    // cudaMemcpy() wrapper
    __forceinline__ inline void cudaMemcpy(void* const dst, const void* const src, const ::size_t count, const ::cudaMemcpyKind kind) {
        // Forward call
        ::cudaMemcpy(dst, src, count, kind);

        // Check for errors
        cudaCheckLastError();
    }

    // cudaMemcpyAsync() wrapper
    __forceinline__ inline void cudaMemcpyAsync(void* const dst, const void* const src, const ::size_t count, const ::cudaMemcpyKind kind, const ::cudaStream_t& stream = cudaStreamDefault) {
        // Forward call
        ::cudaMemcpyAsync(dst, src, count, kind, stream);

        // Check for errors
        cudaCheckLastError();
    }

    // cudaMemset() wrapper
    __forceinline__ inline void cudaMemset(void* const d_ptr, const int value, const ::size_t count) {
        // Forward call
        ::cudaMemset(d_ptr, value, count);

        // Check for errors
        cudaCheckLastError();
    }

    // cudaMemsetAsync() wrapper
    __forceinline__ inline void cudaMemsetAsync(void* const d_ptr, const int value, const ::size_t count, const ::cudaStream_t& stream = cudaStreamDefault) {
        // Forward call
        ::cudaMemsetAsync(d_ptr, value, count, stream);

        // Check for errors
        cudaCheckLastError();
    }

    // cudaLaunchKernel wrapper
    template <auto Kernel, typename... Args>
    __forceinline__ void cudaLaunchKernel(const ::size_t dynamicSMemSize, const int maxBlockSize, const ::size_t numElements, const bool sync, Args&&... args) {
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

    class BumpPtrAllocator {
    public:
        // Default constructor
        BumpPtrAllocator() = default;

        // Copy constructor
        BumpPtrAllocator(const BumpPtrAllocator&) = default;

        // Move constructor
        BumpPtrAllocator(BumpPtrAllocator&&) = default;

        // Copy assignment
        BumpPtrAllocator& operator=(const BumpPtrAllocator&) = default;

        // Move assignment
        BumpPtrAllocator& operator=(BumpPtrAllocator&&) = default;

        // Destructor
        ~BumpPtrAllocator() = default;

        // Get size
        __host__ __device__ __forceinline__ ::size_t getSize() const {
            return size;
        }

        // Get pointer
        __host__ __device__ __forceinline__ ::uintptr_t getPtr() const {
            return ptr;
        }

        // Peak pointer for next allocation
        template <typename T>
        __host__ __device__ __forceinline__ const T* peek() const {
            // Check if not initialized
            if (!ptr) {
                return nullptr;
            }

            // Get type size and type alignment
            constexpr auto typeSize{sizeof(T)};
            constexpr auto typeAlignment{alignof(T)};

            // Align the current pointer to the type alignment
            const auto alignedCurrentPtr{align(ptr, typeAlignment)};

            // Get new size by adding the type size to the aligned size
            const auto newPtr{alignedCurrentPtr + typeSize};

            // Get total growth and check for overflow
            if (const auto totalGrowth{newPtr - ptr}; totalGrowth > size) {
                return nullptr;
            }

            return alignedCurrentPtr;
        }

        // Grow allocator size
        template <typename T>
        __host__ __device__ __forceinline__ void grow(const ::size_t count = 1) {
            // Check if already initialized
            if (ptr) {
                return;
            }

            // Get type size and type alignment
            const auto typeSize{count * sizeof(T)};
            constexpr auto typeAlignment{alignof(T)};

            // Align the current size to the type alignment
            const auto alignedCurrentSize{align(size, typeAlignment)};

            // Get new size by adding the type size to the aligned size
            const auto newSize{alignedCurrentSize + typeSize};

            // Set size to new size
            this->size = newSize;
        }

        // Emplace using constructor and bump pointer
        template <typename T, typename... Args>
        __host__ __device__ __forceinline__ T* emplaceConstruct(Args&&... args) {
            // Get size of type
            constexpr auto typeSize{sizeof(T)};

            // Reserve space
            const auto reserved{reserve<T>(typeSize)};

            // Check if reservation was successful
            if (!reserved) {
                return nullptr;
            }

            // Construct object
            const auto object{new (reserved) T(std::forward<Args>(args)...)};

            return object;
        }

        // Emplace by just reserving space and bump pointer
        template <typename T>
        __host__ __device__ __forceinline__ T* emplaceReserve(const ::size_t count = 1) {
            // Get total size
            const auto totalSize{count * sizeof(T)};

            // Reserve space
            T* const reserved = reserve<T>(totalSize);

            // Check if reservation was successful
            if (!reserved) {
                return nullptr;
            }

            return reserved;
        }

        // Emplace using memcpy and bump pointer
        template <typename T>
        __host__ __device__ __forceinline__ T* emplaceCopy(const T* src, const ::size_t count = 1) {
            // Get total size
            const auto totalSize{count * sizeof(T)};

            // Reserve space
            T* const reserved = reserve<T>(totalSize);

            // Check if reservation was successful and source is valid
            if (!reserved || !src) {
                return nullptr;
            }

            // Copy memory
            ::memcpy(reserved, src, totalSize);

            return reserved;
        }

        // Emplace using memset and bump pointer
        template <typename T>
        __host__ __device__ __forceinline__ T* emplaceSet(const int value, const ::size_t count = 1) {
            // Get total size
            const auto totalSize{count * sizeof(T)};

            // Reserve space
            T* const reserved = reserve<T>(totalSize);

            // Check if reservation was successful
            if (!reserved) {
                return nullptr;
            }

            // Set memory
            ::memset(reserved, value, totalSize);

            return reserved;
        }

        // Emplace using cudaMemcpy and bump pointer
        template <typename T>
        __host__ __device__ __forceinline__ T* cudaEmplaceCopy(const T* src, const ::cudaMemcpyKind kind, const ::size_t count = 1, const bool sync = true, const ::cudaStream_t& stream = cudaStreamDefault) {
            // Get total size
            const auto totalSize{count * sizeof(T)};

            // Reserve space
            T* const reserved = reserve<T>(totalSize);

            // Check if reservation was successful and source is valid
            if (!reserved || !src) {
                return nullptr;
            }

            // Copy memory
            if (sync) {
                cudaMemcpy(reserved, src, totalSize, kind);
            }
            else {
                cudaMemcpyAsync(reserved, src, totalSize, kind, stream);
            }

            return reserved;
        }

        // Emplace using cudaMemset and bump pointer
        template <typename T>
        __host__ __device__ __forceinline__ T* cudaEmplaceSet(const int value, const ::size_t count = 1, const bool sync = true, const ::cudaStream_t& stream = cudaStreamDefault) {
            // Get total size
            const auto totalSize{count * sizeof(T)};

            // Reserve space
            T* const reserved = reserve<T>(totalSize);

            // Check if reservation was successful
            if (!reserved) {
                return nullptr;
            }

            // Set memory
            if (sync) {
                cudaMemset(reserved, value, totalSize);
            }
            else {
                cudaMemsetAsync(reserved, value, totalSize, stream);
            }

            return reserved;
        }

        // Initialize using host memory
        __host__ __forceinline__ void initHostMem() {
            // Malloc memory and set pointer
            this->ptr = reinterpret_cast<::uintptr_t>(::malloc(size));
        }

        // Initialize using host pinned memory
        __host__ __forceinline__ void initHostPinnedMem() {
            // Cuda malloc host
            void* pinned_ptr{nullptr};
            cudaMallocHost(&pinned_ptr, size);

            // Set pointer
            this->ptr = reinterpret_cast<::uintptr_t>(pinned_ptr);
        }

        // Initialize using device global memory
        __host__ __device__ __forceinline__ void initCudaGMem() {
            // Cuda malloc memory
            void* d_ptr{nullptr};
            cudaMalloc(&d_ptr, size);

            // Set pointer
            this->ptr = reinterpret_cast<::uintptr_t>(d_ptr);
        }

        // Initialize asynchronously using device global memory
        __host__ __device__ __forceinline__ void initCudaGMemAsync(const cudaStream_t& stream = cudaStreamDefault) {
            // Cuda malloc memory
            void* d_ptr{nullptr};
            cudaMallocAsync(&d_ptr, size, stream);

            // Set pointer
            this->ptr = reinterpret_cast<::uintptr_t>(d_ptr);
        }

        // Initialize using device shared memory
        __device__ __forceinline__ void initCudaSMem(const ::uintptr_t sharedMemPtr, const ::size_t sharedMemSize) {
            // Set pointer to the already allocated shared memory
            this->ptr = sharedMemPtr;

            // Set size to the allocated shared memory size
            this->size = sharedMemSize;
        }

    private:
        // Bump pointer allocator implementation
        ::size_t size{0};
        ::uintptr_t ptr{0};

        // Align size according to the given alignment
        __host__ __device__ __forceinline__ static ::size_t align(const ::size_t size, const ::size_t alignment) {
            return (size + alignment - 1) & ~(alignment - 1);
        }

        // Reserve total bytes for a given type, according to its alignment
        template <typename T>
        __host__ __device__ __forceinline__ T* reserve(const ::size_t totalBytes) {
            // Check if not initialized or total bytes is zero
            if (!ptr || totalBytes == 0) {
                return nullptr;
            }

            // Get type alignment
            constexpr auto typeAlignment{alignof(T)};

            // Align the current pointer to the type alignment
            const auto alignedCurrentPtr{align(ptr, typeAlignment)};

            // Get new size by adding the type size to the aligned size
            const auto newPtr{alignedCurrentPtr + totalBytes};

            // Get the total growth
            const auto totalGrowth{newPtr - ptr};

            // Check for overflow
            if (totalGrowth > size) {
                return nullptr;
            }

            // Decrease size
            this->size -= totalGrowth;

            // Bump pointer
            this->ptr = newPtr;

            return alignedCurrentPtr;
        }
    };
} // cuSGA::KernelUtils

#endif //CUSGA_KERNELUTILS_CUH
