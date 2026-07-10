#ifndef CUSGA_FRONTIER_CUH
#define CUSGA_FRONTIER_CUH
#include <cuda/atomic>

#include "DoubleBuffer.cuh"

namespace cuSGA {
    class Frontier {
    public:
        // Create frontier
        __host__ static Frontier* create(::uint64_t size);

        // Move frontier to device
        __host__ Frontier* copyToDevice();
        // Free frontier
        __host__ void free() const;

        // Get frontier size
        __host__ __device__ ::uint64_t getSize() const;
        // Check if frontier is empty
        __host__ __device__ bool isEmpty() const;
        // Check if device frontier is empty
        __host__ bool isEmptySync();
        // Get node index for a given index
        __host__ __device__ ::uint64_t getNodeIndex(::uint64_t idx) const;
        // Get device instance
        __host__ __device__ Frontier* getDeviceInstance() const;

        // Set frontier size
        __host__ __device__ void setSize(::uint64_t size);
        // Insert a node into the current frontier without queueing
        __host__ __device__ void insertWithoutQueueing(::uint64_t nodeIdx) const;
        // Atomically insert a given node into the queue and grow its size
        __host__ __device__ void atomicInsertAndGrow(::uint64_t nodeIdx);
        // Swap from queue to next frontier
        __host__ __device__ void swapToQueue();
        // Swap from device queue to next device frontier
        __host__ void swapToQueueSync() const;
        // Empty frontier
        __host__ __device__ void empty();
        // Empty device frontier
        __host__ void emptySync() const;

    private:
        // Frontier implementation
        ::uint64_t currentSize{0};
        ::cuda::atomic<::uint64_t, ::cuda::thread_scope_device> alternateSize{0};
        DoubleBuffer<::uint64_t>* doubleBuffer{nullptr};
        ::cuda::atomic<bool, ::cuda::thread_scope_device>* isInQueue{nullptr};
        Frontier* d_instance{nullptr};

        // Default constructor
        Frontier() = default;
        // Frontier constructor
        __host__ __device__ Frontier(::uint64_t currentSize, ::uint64_t alternateSize, DoubleBuffer<::uint64_t>* doubleBuffer, ::cuda::atomic<bool, ::cuda::thread_scope_device>* isInQueue, Frontier* d_instance = nullptr);
    };
} // cuSGA

#endif //CUSGA_FRONTIER_CUH
