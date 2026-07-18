#ifndef CUSGA_FRONTIER_CUH
#define CUSGA_FRONTIER_CUH
#include <cuda/atomic>

#include "DoubleBuffer.cuh"

namespace cuSGA {
    // Frontier
    class Frontier {
    public:
        // Grow allocator using the expected buffers size
        __host__ __device__ __forceinline__ static void growBuffers(KernelUtils::BumpPtrAllocator* const allocator, const ::size_t size) {
            // Grow size for double buffer
            DoubleBuffer<::size_t>::growBuffers(allocator, size);

            // Grow size for isInQueue
            allocator->grow<::std::remove_pointer_t<decltype(isInQueue)>>(size);
        }

        // Default constructor
        Frontier() = default;
        // Parameterized constructor
        __host__ Frontier(::size_t size, bool ownsInstance, Frontier* pinned_instanceOptional = nullptr, KernelUtils::BumpPtrAllocator* allocatorOptional = nullptr);
        // Copy constructor
        Frontier(const Frontier& other) = default;
        // Move constructor
        Frontier(Frontier&& other) = default;
        // Copy assignment
        Frontier& operator=(const Frontier& other) = default;
        // Move assignment
        Frontier& operator=(Frontier&& other) = default;
        // Destructor
        ~Frontier() = default;

        // Move frontier to device
        __host__ Frontier copyToDevice(Frontier* d_instanceOptional = nullptr, KernelUtils::BumpPtrAllocator* allocatorOptional = nullptr);
        // Free frontier
        __host__ void free() const;

        // Get frontier size
        __host__ __device__ __forceinline__ ::size_t getSize() const {
            return currentSize;
        }

        // Get frontier max size
        __host__ __device__ __forceinline__ ::size_t getMaxSize() const {
            return doubleBuffer.getSize();
        }

        // Check if frontier is empty
        __host__ __device__ __forceinline__ bool isEmpty() const {
            return currentSize == 0;
        }

        // Check if device frontier is empty
        __host__ __forceinline__ bool h2d_isEmpty() const {
            // Copy back current size from device
            if (d_instance) {
                CUDA_CHECK(::cudaMemcpy(&pinned_instance->currentSize, &d_instance->currentSize, sizeof(currentSize), ::cudaMemcpyDeviceToHost));
            }

            return pinned_instance->currentSize == 0;
        }

        // Get node index for a given index
        __host__ __device__ __forceinline__ ::size_t getValue(const ::size_t idx) const {
            return doubleBuffer.current()[idx];
        }

        // Get pinned instance
        __host__ __device__ __forceinline__ Frontier* getPinnedInstance() const {
            return pinned_instance;
        }

        // Get device instance
        __host__ __device__ __forceinline__ Frontier* getDeviceInstance() const {
            return d_instance;
        }

        // Get buffers root
        __host__ __device__ __forceinline__ void* getBuffersRoot() const {
            return doubleBuffer.getBuffersRoot();
        }

        // Set frontier size
        __host__ __device__ __forceinline__ void setSize(const ::size_t size) {
            this->currentSize = size;
        }

        // Set device frontier size
        __host__ __forceinline__ void h2d_setSize(const ::size_t size) const {
            pinned_instance->currentSize = size;
            CUDA_CHECK(::cudaMemcpyAsync(&d_instance->currentSize, &pinned_instance->currentSize, sizeof(currentSize), ::cudaMemcpyHostToDevice, cudaStreamDefault));
        }

        // Set device frontier size
        __device__ __forceinline__ void d2d_setSize(const ::size_t size) const {
            pinned_instance->currentSize = size;
        }

        // Grow frontier queue size
        __host__ __device__ __forceinline__ void growQueueSize() {
            this->alternateSize += 1;
        }

        // Grow device frontier queue size
        __host__ __forceinline__ void h2d_growQueueSize() const {
            pinned_instance->alternateSize += 1;
            CUDA_CHECK(::cudaMemcpyAsync(&d_instance->alternateSize, &pinned_instance->alternateSize, sizeof(alternateSize), ::cudaMemcpyHostToDevice, cudaStreamDefault));
        }

        // Grow device frontier queue size
        __device__ __forceinline__ void d2d_growQueueSize() const {
            d_instance->alternateSize += 1;
        }

        // Insert a node into the current frontier without queueing
        __host__ __device__ __forceinline__ void insertWithoutQueueing(const ::size_t nodeIdx) const {
            doubleBuffer.current()[nodeIdx] = nodeIdx;
        }

        // Atomically insert a given node into the queue and grow its size
        __host__ __device__ __forceinline__ void atomicInsertAndGrow(const ::size_t nodeIdx) {
            // Insert atomically in queue
            if (const auto wasInQueue{isInQueue[nodeIdx].exchange(true, ::cuda::memory_order_relaxed)}; !wasInQueue) {
                // Insert node into next frontier
                doubleBuffer.alternate()[alternateSize] = nodeIdx;

                // Grow queue size
                this->alternateSize += 1;
            }
        }

        // Atomically insert a given node into the queue and grow its size
        __device__ __forceinline__ void d2d_atomicInsertAndGrow(const ::size_t nodeIdx) const {
            // Insert atomically in queue
            if (const auto wasInQueue{isInQueue[nodeIdx].exchange(true, ::cuda::memory_order_relaxed)}; !wasInQueue) {
                // Insert node into next frontier
                doubleBuffer.alternate()[alternateSize] = nodeIdx;

                // Grow queue size
                d_instance->alternateSize += 1;
            }
        }

        // Swap from queue to next frontier
        __host__ __device__ __forceinline__ void swapToQueue() {
            // Swap buffers
            this->doubleBuffer.swap();

            // Swap sizes
            this->currentSize = alternateSize;
            this->alternateSize = 0;

            // Clear isInQueue
            ::memset(isInQueue, false, doubleBuffer.getSize() * sizeof(isInQueue[0]));
        }

        // Swap from device queue to next device frontier
        __host__ __forceinline__ void h2d_swapToQueue() const {
            if (d_instance) {
                // Swap buffers
                doubleBuffer.h2d_swap();

                // Swap sizes
                pinned_instance->currentSize = pinned_instance->alternateSize;
                pinned_instance->alternateSize = 0;

                // Clear isInQueue
                CUDA_CHECK(::cudaMemsetAsync(pinned_instance->isInQueue, false, doubleBuffer.getSize() * sizeof(isInQueue[0]), cudaStreamDefault));

                // Update device instance
                CUDA_CHECK(::cudaMemcpyAsync(d_instance, pinned_instance, sizeof(Frontier), ::cudaMemcpyHostToDevice, cudaStreamDefault));
            }
        }

        // Empty frontier
        __host__ __device__ __forceinline__ void empty() {
            // Clear (virtually) the buffers
            this->currentSize = 0;
            this->alternateSize = 0;

            // Clear isInQueue
            ::memset(isInQueue, false, doubleBuffer.getSize() * sizeof(isInQueue[0]));
        }

        // Empty device frontier
        __host__ __forceinline__ void h2d_empty() const {
            if (d_instance) {
                // Clear (virtually) the device buffers
                pinned_instance->currentSize = 0;
                pinned_instance->alternateSize = 0;

                // Clear isInQueue
                CUDA_CHECK(::cudaMemsetAsync(pinned_instance->isInQueue, 0, doubleBuffer.getSize() * sizeof(isInQueue[0]), cudaStreamDefault));

                // Update device instance
                CUDA_CHECK(::cudaMemcpyAsync(d_instance, pinned_instance, sizeof(Frontier), ::cudaMemcpyHostToDevice, cudaStreamDefault));
            }
        }

    private:
        // Frontier implementation
        // NOTE: Uses pinned memory and linearized memory layout on the device memory
        ::size_t currentSize{0};
        ::size_t alternateSize{0};
        DoubleBuffer<::size_t> doubleBuffer{};
        ::cuda::atomic<bool, ::cuda::thread_scope_device>* isInQueue{nullptr};
        bool ownsInstance{false};
        Frontier* pinned_instance{nullptr};
        Frontier* d_instance{nullptr};

        // Parameterized constructor
        __host__ __device__ __forceinline__ Frontier(const ::size_t currentSize, const ::size_t alternateSize, const DoubleBuffer<::size_t>& doubleBuffer, ::cuda::atomic<bool, ::cuda::thread_scope_device>* isInQueue, const bool ownsInstance, Frontier* const pinned_instance = nullptr, Frontier* const d_instance = nullptr) : currentSize(currentSize), alternateSize(alternateSize), doubleBuffer(doubleBuffer), isInQueue(isInQueue), ownsInstance(ownsInstance), pinned_instance(pinned_instance), d_instance(d_instance) {}
    };
} // cuSGA

#endif //CUSGA_FRONTIER_CUH
