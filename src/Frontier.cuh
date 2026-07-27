#ifndef CUSGA_FRONTIER_CUH
#define CUSGA_FRONTIER_CUH
#include "DoubleBuffer.cuh"
#include "PangenomeGraph.cuh"

namespace cuSGA {
    // Define queue pack type
    using queuePack_t = ::uint32_t;

    // Frontier
    class Frontier {
    public:
        // Frontier related constants
        static constexpr ::uint8_t BIT_SIZE{1};
        static constexpr queuePack_t BITMASK{(1u << BIT_SIZE) - 1};
        static constexpr ::uint8_t PACKING_FACTOR{(sizeof(queuePack_t) * Utils::BYTE_SIZE) / BIT_SIZE};
        static constexpr ::uint8_t PACK_SHIFT{::std::countr_zero(PACKING_FACTOR)};

        // Grow allocator using the expected buffers size
        __host__ __device__ __forceinline__ static void growBuffers(KernelUtils::BumpPtrAllocator* const allocator, const nodeSize_t size) {
            // Grow size for double buffer
            DoubleBuffer<nodeSize_t>::growBuffers(allocator, size);

            // Grow size for isInQueue
            const auto numChunks{(size + PACKING_FACTOR - 1) >> PACK_SHIFT};
            allocator->grow<::std::remove_reference_t<decltype(isInQueue[0])>>(numChunks);
        }

        // Default constructor
        Frontier() = default;

        // Parameterized constructor
        __host__ __device__ __forceinline__ Frontier(const nodeSize_t currentSize, const nodeSize_t alternateSize, const DoubleBuffer<nodeSize_t>& __restrict__ doubleBuffer, queuePack_t* const isInQueue) : currentSize{currentSize}, alternateSize{alternateSize}, doubleBuffer{doubleBuffer}, isInQueue{isInQueue} {}

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

        // Get frontier size
        __host__ __device__ __forceinline__ nodeSize_t getSize() const {
            return currentSize;
        }

        // Get queue size
        __host__ __device__ __forceinline__ nodeSize_t getQueueSize() const {
            return alternateSize;
        }

        // Get frontier max size
        __host__ __device__ __forceinline__ nodeSize_t getMaxSize() const {
            return doubleBuffer.getSize();
        }

        // Check if frontier is empty
        __host__ __device__ __forceinline__ bool isEmpty() const {
            return currentSize == 0;
        }

        // Get frontier values
        __host__ __device__ __forceinline__ const nodeSize_t* __restrict__ getValues() const {
            return doubleBuffer.current();
        }

        // Get node index for a given index
        __host__ __device__ __forceinline__ nodeSize_t getValue(const nodeSize_t frontierIdx) const {
            return doubleBuffer.current()[frontierIdx];
        }

        // Get isInQueue array
        __host__ __device__ __forceinline__ queuePack_t* getIsInQueue() const {
            return isInQueue;
        }

        // Get buffers root
        __host__ __device__ __forceinline__ void* __restrict__ getBuffersRoot() const {
            return doubleBuffer.getBuffersRoot();
        }

        // Check if given node is in queue
        __host__ __device__ __forceinline__ bool isNodeInQueue(const nodeSize_t nodeIdx) const {
            // Get pack index and bitmask
            const auto chunkIdx = nodeIdx >> PACK_SHIFT;
            const auto bitmask = BITMASK << (nodeIdx & (PACKING_FACTOR - 1));

            return (isInQueue[chunkIdx] & bitmask) != 0;
        }

        // Set frontier size
        __host__ __device__ __forceinline__ void setSize(const nodeSize_t size) {
            this->currentSize = size;
        }

        // Grow frontier queue size
        __host__ __device__ __forceinline__ void growQueueSize(const nodeSize_t count = 1) {
            this->alternateSize += count;
        }

        // Insert a node into the current frontier without queueing
        __host__ __device__ __forceinline__ void insertWithoutQueueing(const nodeSize_t nodeIdx, const nodeSize_t frontierIdx) const {
            doubleBuffer.current()[frontierIdx] = nodeIdx;
        }

        // Insert a node into the current frontier queue
        __host__ __device__ __forceinline__ void insertInQueue(const nodeSize_t nodeIdx, const nodeSize_t queueIdx) const {
            // Update double buffer
            doubleBuffer.alternate()[queueIdx] = nodeIdx;

            // Get pack index and bitmask
            const auto chunkIdx = nodeIdx >> PACK_SHIFT;
            const auto bitmask = BITMASK << (nodeIdx & (PACKING_FACTOR - 1));

            // Perform OR to set the bit
            isInQueue[chunkIdx] |= bitmask;
        }

        // Atomically insert a node into the current frontier queue
        __device__ __forceinline__ void atomicInsertNodeInQueue(const nodeSize_t nodeIdx, const nodeSize_t queueIdx) const {
            // Update double buffer
            doubleBuffer.alternate()[queueIdx] = nodeIdx;

            // Get pack index and bitmask
            const auto chunkIdx = nodeIdx >> PACK_SHIFT;
            const auto bitmask = BITMASK << (nodeIdx & (PACKING_FACTOR - 1));

            // Perform atomic OR to set bit
            ::atomicOr(isInQueue + chunkIdx, bitmask);
        }

        // Swap from queue to next frontier
        __host__ __device__ __forceinline__ void swap() {
            // Swap buffers
            this->doubleBuffer.swap();

            // Swap sizes
            this->currentSize = alternateSize;
            this->alternateSize = 0;
        }

        // Shuffle object from the given lane, with the given mask
        __device__ __forceinline__ void shuffle_sync(const unsigned mask, const int srcLaneIdx) {
            this->currentSize = ::__shfl_sync(mask, currentSize, srcLaneIdx);
            this->alternateSize = ::__shfl_sync(mask, alternateSize, srcLaneIdx);
            this->doubleBuffer.shuffle_sync(mask, srcLaneIdx);
            this->isInQueue = reinterpret_cast<decltype(isInQueue)>(::__shfl_sync(mask, reinterpret_cast<unsigned long long>(isInQueue), srcLaneIdx));
        }

    private:
        // Frontier implementation
        nodeSize_t currentSize{0};
        nodeSize_t alternateSize{0};
        DoubleBuffer<nodeSize_t> doubleBuffer{};
        queuePack_t* isInQueue{nullptr};
    };
} // cuSGA

#endif //CUSGA_FRONTIER_CUH
