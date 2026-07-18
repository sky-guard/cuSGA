#ifndef CUSGA_DOUBLEBUFFER_CUH
#define CUSGA_DOUBLEBUFFER_CUH
#include "KernelUtils.cuh"

namespace cuSGA {
    // Double buffer
    template <typename T>
    class DoubleBuffer {
    public:
        // Double buffer related constants
        static constexpr ::size_t NUM_DOUBLE_BUFFERS{2};

        // Grow allocator using the expected buffers size
        __host__ __device__ __forceinline__ static void growBuffers(KernelUtils::BumpPtrAllocator* const allocator, const ::size_t size) {
            // Grow size for buffers
            allocator->grow<T>(NUM_DOUBLE_BUFFERS * size);
        }

        // Default constructor
        DoubleBuffer() = default;

        // Parameterized constructor
        __host__ DoubleBuffer(const ::size_t size, const bool ownsInstance,  DoubleBuffer* const pinned_instanceOptional = nullptr, KernelUtils::BumpPtrAllocator* const allocatorOptional = nullptr) : DoubleBuffer(size, {nullptr}, 0, ownsInstance, pinned_instanceOptional) {
            // Get allocator
            KernelUtils::BumpPtrAllocator* allocator{allocatorOptional};
            KernelUtils::BumpPtrAllocator allocatorInstance{};
            if (!allocator) {
                allocator = &allocatorInstance;
            }

            // Grow allocator
            if (ownsInstance) {
                allocator->grow<DoubleBuffer>();
                growBuffers(allocator, size);
            }

            // Initialize allocator
            if (ownsInstance) {
                allocator->initHostPinnedMem();
            }

            // Emplace buffers
            if (ownsInstance) {
                this->pinned_instance = allocator->emplaceReserve<DoubleBuffer>();
            }
            const auto basePtr{allocator->emplaceReserve<T>(NUM_DOUBLE_BUFFERS * size)};
#pragma unroll
            for (::size_t i{0}; i < NUM_DOUBLE_BUFFERS; ++i) {
                this->buffers[i] = basePtr + i * (size * sizeof(T));
            }
        }

        // Copy constructor
        DoubleBuffer(const DoubleBuffer& other) = default;

        // Move constructor
        DoubleBuffer(DoubleBuffer&& other) = default;

        // Copy assignment
        DoubleBuffer& operator=(const DoubleBuffer& other) = default;

        // Move assignment
        DoubleBuffer& operator=(DoubleBuffer&& other) = default;

        // Destructor
        ~DoubleBuffer() = default;

        // Move double buffer to device
        __host__ DoubleBuffer copyToDevice(DoubleBuffer* const d_instanceOptional = nullptr, KernelUtils::BumpPtrAllocator* const allocatorOptional = nullptr) {
            // Check if device instance already exists for this double buffer
            if (d_instance) {
                throw ::std::runtime_error{"Device instance already exists for this Double Buffer!"};
            }

            // Get allocator
            KernelUtils::BumpPtrAllocator* allocator{allocatorOptional};
            KernelUtils::BumpPtrAllocator allocatorInstance{};
            if (!allocator) {
                allocator = &allocatorInstance;
            }

            // Grow allocator
            if (ownsInstance) {
                allocator->grow<DoubleBuffer>();
                growBuffers(allocator, size);
            }

            // Initialize allocator
            if (ownsInstance) {
                allocator->initCudaGMem();
            }

            // Reserve instance
            if (ownsInstance) {
                this->d_instance = allocator->emplaceReserve<DoubleBuffer>();
            }
            else {
                this->d_instance = d_instanceOptional;
            }

            // Emplace buffers
            const auto d_buffersBase{allocator->cudaEmplaceCopy<T>(buffers[0], ::cudaMemcpyHostToDevice, NUM_DOUBLE_BUFFERS * size, false, cudaStreamDefault)};
            T* d_buffers[NUM_DOUBLE_BUFFERS]{nullptr};
#pragma unroll
            for (::size_t i{0}; i < NUM_DOUBLE_BUFFERS; ++i) {
                this->buffers[i] = d_buffersBase + i * (size * sizeof(T));
            }

            // Create temporary host instance holding the device pointers
            const DoubleBuffer d_doubleBuffer{size, d_buffers, selector, ownsInstance, pinned_instance, d_instance};

            // Emplace instance
            if (ownsInstance) {
                *pinned_instance = d_doubleBuffer;
                CUDA_CHECK(::cudaMemcpyAsync(d_instance, pinned_instance, sizeof(DoubleBuffer), ::cudaMemcpyHostToDevice, cudaStreamDefault));
            }

            return d_doubleBuffer;
        }

        // Free double buffer
        __host__ void free() const {
            // Free device memory if present
            if (d_instance) {
                if (ownsInstance) {
                    CUDA_CHECK(::cudaFreeAsync(d_instance, cudaStreamDefault));
                }
                else {
                    CUDA_CHECK(::cudaFreeAsync(pinned_instance->getBuffersRoot(), cudaStreamDefault));
                }
            }

            // Free host memory
            if (ownsInstance) {
                CUDA_CHECK(::cudaFreeHost(pinned_instance));
            }
            else {
                CUDA_CHECK(::cudaFreeHost(getBuffersRoot()));
            }
        }

        // Get size
        __host__ __device__ __forceinline__ ::size_t getSize() const {
            return size;
        }

        // Get current buffer
        __host__ __device__ __forceinline__ T* current() const {
            return buffers[selector];
        }

        // Get alternate buffer
        __host__ __device__ __forceinline__ T* alternate() const {
            return buffers[selector ^ 1];
        }

        // Get pinned instance
        __host__ __device__ __forceinline__ DoubleBuffer* getPinnedInstance() const {
            return pinned_instance;
        }

        // Get device instance
        __host__ __device__ __forceinline__ DoubleBuffer* getDeviceInstance() const {
            return d_instance;
        }

        // Get buffer root
        __host__ __device__ __forceinline__ void* getBuffersRoot() const {
            return buffers[0];
        }

        // Swap buffers
        __host__ __device__ __forceinline__ void swap() {
            this->selector ^= 1;
        }

        // Swap device buffers
        __host__ __forceinline__ void h2d_swap() const {
            if (d_instance) {
                // Update pinned instance
                pinned_instance->selector ^= 1;

                // Update device instance asynchronously
                CUDA_CHECK(::cudaMemcpyAsync(&d_instance->selector, &pinned_instance->selector, sizeof(selector), ::cudaMemcpyHostToDevice, cudaStreamDefault));
            }
        }

        // Non-const version of subscription operator for assignment and modification
        __host__ __device__ __forceinline__ T& operator[](const ::size_t idx) {
            return buffers[selector][idx];
        }

        // Const version of subscription operator for read-only access
        __host__ __device__ __forceinline__ const T& operator[](const ::size_t idx) const {
            return buffers[selector][idx];
        }

    private:
        // Double buffer implementation
        // NOTE: Uses pinned memory and linearized memory layout on the device memory
        ::size_t size{0};
        T* buffers[NUM_DOUBLE_BUFFERS]{nullptr};
        ::size_t selector{0};
        bool ownsInstance{false};
        DoubleBuffer* pinned_instance{nullptr};
        DoubleBuffer* d_instance{nullptr};

        // Parameterized constructor
        __host__ __device__ __forceinline__ DoubleBuffer(const ::size_t size, T* const (& buffers)[NUM_DOUBLE_BUFFERS], const ::size_t selector, const bool ownsInstance, DoubleBuffer* const pinned_instance = nullptr, DoubleBuffer* const d_instance = nullptr) : size(size), selector(selector), ownsInstance(ownsInstance), pinned_instance(pinned_instance), d_instance(d_instance) {
            // Set buffers
#pragma unroll
            for (::size_t i{0}; i < NUM_DOUBLE_BUFFERS; ++i) {
                this->buffers[i] = buffers[i];
            }
        }
    };
} // cuSGA

#endif //CUSGA_DOUBLEBUFFER_CUH
