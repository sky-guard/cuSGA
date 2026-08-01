#ifndef CUSGA_PANGENOMEGRAPH_CUH
#define CUSGA_PANGENOMEGRAPH_CUH
#include <string>

#include "DNABase.cuh"
#include "PackedDNASequence.cuh"

namespace cuSGA {
    // Define node size type
    using nodeSize_t = targetSize_t;

    // Define edge size type
    using edgeSize_t = targetSize_t;

    // Pangenome graph
    class PangenomeGraph {
    public:
        // Grow allocator using the expected buffers size
        __host__ __device__ __forceinline__ static void growBuffers(KernelUtils::BumpPtrAllocator* allocator, const nodeSize_t numNodes, const edgeSize_t numEdges) {
            // Grow size for sequence
            PackedDNASequence::growBuffers(allocator, numNodes);

            // Grow size for column values
            allocator->grow<::std::remove_reference_t<decltype(columnValues[0])>>(numEdges);

            // Grow size for row offsets
            allocator->grow<::std::remove_reference_t<decltype(rowOffsets[0])>>(numNodes + 1);
        }

        // Default constructor
        PangenomeGraph() = default;
        // Parameterized constructor
        __host__ PangenomeGraph(const ::std::string& fileName, bool ownsInstance, PangenomeGraph* pinned_instanceOptional = nullptr, KernelUtils::BumpPtrAllocator* allocatorOptional = nullptr);
        // Parameterized constructor
        __host__ PangenomeGraph(const ::std::string& fileName, ::std::ifstream* file, bool ownsInstance, ::std::optional<nodeSize_t> numNodesOptional = ::std::nullopt, ::std::optional<edgeSize_t> numEdgesOptional = ::std::nullopt, PangenomeGraph* pinned_instanceOptional = nullptr, KernelUtils::BumpPtrAllocator* allocatorOptional = nullptr);

        // Pangenome graph constructor
        __host__ __device__ __forceinline__ PangenomeGraph(const edgeSize_t numEdges, const PackedDNASequence& baseValues, nodeSize_t* const columnValues, edgeSize_t* const rowOffsets, const bool ownsInstance, PangenomeGraph* const pinned_instance = nullptr, PangenomeGraph* const d_instance = nullptr) : numEdges{numEdges}, baseValues{baseValues}, columnValues{columnValues}, rowOffsets{rowOffsets}, ownsInstance{ownsInstance}, pinned_instance{pinned_instance}, d_instance{d_instance} {}

        // Copy constructor
        PangenomeGraph(const PangenomeGraph& other) = default;
        // Move constructor
        PangenomeGraph(PangenomeGraph&& other) = default;
        // Copy assignment
        PangenomeGraph& operator=(const PangenomeGraph& other) = default;
        // Move assignment
        PangenomeGraph& operator=(PangenomeGraph&& other) = default;
        // Destructor
        ~PangenomeGraph() = default;

        // Move pangenome graph to device
        __host__ PangenomeGraph copyToDevice(PangenomeGraph* d_instanceOptional = nullptr, KernelUtils::BumpPtrAllocator* allocatorOptional = nullptr);
        // Free pangenome graph
        __host__ void free() const;

        // Get number of nodes in the graph
        __host__ __device__ __forceinline__ nodeSize_t getNumNodes() const {
            return baseValues.getNumBases();
        }

        // Get number of edges in the graph
        __host__ __device__ __forceinline__ edgeSize_t getNumEdges() const {
            return numEdges;
        }

        // Get DNA base values for the graph
        __host__ __device__ __forceinline__ const PackedDNASequence& getBaseValues() const {
            return baseValues;
        }

        // Get DNA base value for a given node index
        __host__ __device__ __forceinline__ DNABase getDNABase(const nodeSize_t nodeIdx) const {
            return baseValues[nodeIdx];
        }

        // Get neighbor values
        __host__ __device__ __forceinline__ nodeSize_t* getNeighborValues() const {
            return columnValues;
        }

        // Get neighbor for a given edge index
        __host__ __device__ __forceinline__ nodeSize_t getNeighbor(const edgeSize_t edgeIdx) const {
            return columnValues[edgeIdx];
        }

        // Get neighbor offsets
        __host__ __device__ __forceinline__ edgeSize_t* getNeighborOffsets() const {
            return rowOffsets;
        }

        // Get neighbors offset for a given node index
        __host__ __device__ __forceinline__ edgeSize_t getNeighborsOffset(const nodeSize_t nodeIdx) const {
            // Get row offset for current node
            return rowOffsets[nodeIdx];
        }

        // Get number of neighbors for a given node index
        __host__ __device__ __forceinline__ nodeSize_t getNumNeighbors(const nodeSize_t nodeIdx) const {
            return rowOffsets[nodeIdx + 1] - rowOffsets[nodeIdx];
        }

        // Get pinned instance
        __host__ __device__ __forceinline__ PangenomeGraph* getPinnedInstance() const {
            return pinned_instance;
        }

        // Get device instance
        __host__ __device__ __forceinline__ PangenomeGraph* getDeviceInstance() const {
            return d_instance;
        }

        // Get buffer root
        __host__ __device__ __forceinline__ void* getBuffersRoot() const {
            return baseValues.getBuffersRoot();
        }

        // Shuffle object from the given lane, with the given mask
        __device__ __forceinline__ void shuffle_sync(const unsigned mask, const int srcLaneIdx) {
            this->numEdges = ::__shfl_sync(mask, numEdges, srcLaneIdx);
            this->baseValues.shuffle_sync(mask, srcLaneIdx);
            this->columnValues = reinterpret_cast<decltype(columnValues)>(::__shfl_sync(mask, reinterpret_cast<unsigned long long>(columnValues), srcLaneIdx));
            this->rowOffsets = reinterpret_cast<decltype(rowOffsets)>(::__shfl_sync(mask, reinterpret_cast<unsigned long long>(rowOffsets), srcLaneIdx));
            this->ownsInstance = ::__shfl_sync(mask, ownsInstance, srcLaneIdx);
            this->pinned_instance = reinterpret_cast<decltype(pinned_instance)>(::__shfl_sync(mask, reinterpret_cast<unsigned long long>(pinned_instance), srcLaneIdx));
            this->d_instance = reinterpret_cast<decltype(d_instance)>(::__shfl_sync(mask, reinterpret_cast<unsigned long long>(d_instance), srcLaneIdx));
        }

    private:
        // CSR pangenome graph implementation
        // NOTE: Uses pinned memory and linearized memory layout on the device memory
        edgeSize_t numEdges{0};
        PackedDNASequence baseValues{};
        nodeSize_t* columnValues{nullptr};
        edgeSize_t* rowOffsets{nullptr};
        bool ownsInstance{false};
        PangenomeGraph* pinned_instance{nullptr};
        PangenomeGraph* d_instance{nullptr};
    };
} // cuSGA

#endif //CUSGA_PANGENOMEGRAPH_CUH
