#ifndef CUSGA_PANGENOMEGRAPH_CUH
#define CUSGA_PANGENOMEGRAPH_CUH
#include <string>

#include "DNABase.cuh"
#include "PackedDNASequence.cuh"

namespace cuSGA {
    class PangenomeGraph {
    public:
        // Grow allocator using the expected buffers size
        __host__ __device__ __forceinline__ static void growBuffers(KernelUtils::BumpPtrAllocator* allocator, const ::size_t numNodes, const ::size_t numEdges) {
            // Grow size for sequence
            PackedDNASequence::growBuffers(allocator, numNodes);

            // Grow size for column values
            allocator->grow<::std::remove_pointer_t<decltype(columnValues)>>(numEdges);

            // Grow size for row offsets
            allocator->grow<::std::remove_pointer_t<decltype(rowOffsets)>>(numNodes + 1);
        }

        // Default constructor
        PangenomeGraph() = default;
        // Parameterized constructor
        __host__ PangenomeGraph(const ::std::string& fileName, bool ownsInstance, PangenomeGraph* pinned_instanceOptional = nullptr, KernelUtils::BumpPtrAllocator* allocatorOptional = nullptr);
        // Parameterized constructor
        __host__ PangenomeGraph(const ::std::string& fileName, ::std::ifstream* file, bool ownsInstance, ::std::optional<::size_t> numNodesOptional = ::std::nullopt, ::std::optional<::size_t> numEdgesOptional = ::std::nullopt, PangenomeGraph* pinned_instanceOptional = nullptr, KernelUtils::BumpPtrAllocator* allocatorOptional = nullptr);
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
        __host__ PangenomeGraph* copyToDevice(PangenomeGraph* d_instanceOptional = nullptr, KernelUtils::BumpPtrAllocator* allocatorOptional = nullptr);
        // Free pangenome graph
        __host__ void free() const;

        // Get number of nodes in the graph
        __host__ __device__ __forceinline__ ::size_t getNumNodes() const {
            return baseValues.getNumBases();
        }

        // Get number of edges in the graph
        __host__ __device__ __forceinline__ ::size_t getNumEdges() const {
            return numEdges;
        }

        // Get DNA base values for the graph
        __host__ __device__ __forceinline__ const PackedDNASequence& getBaseValues() const {
            return baseValues;
        }

        // Get DNA base value for a given node index
        __host__ __device__ __forceinline__ DNABase getDNABase(const ::size_t nodeIdx) const {
            return baseValues[nodeIdx];
        }

        // Get neighbors for a given node index
        __host__ __device__ __forceinline__ const ::size_t* getNeighbors(const ::size_t nodeIdx) const {
            // Get row offset for current node
            const auto rowOffset{rowOffsets[nodeIdx]};

            // Compute address for neighbors
            const auto neighbors{&columnValues[rowOffset]};

            return neighbors;
        }

        // Get number of neighbors for a given node index
        __host__ __device__ __forceinline__ ::size_t getNumNeighbors(const ::size_t nodeIdx) const {
            // Get row offsets for current node and next in the sequence
            const auto rowOffset{rowOffsets[nodeIdx]};
            const auto nextRowOffset{rowOffsets[nodeIdx + 1]};

            // Compute number of neighbors as the difference between the two
            const auto numNeighbors{nextRowOffset - rowOffset};

            return numNeighbors;
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

    private:
        // CSR pangenome graph implementation
        // NOTE: Uses pinned memory and linearized memory layout on the device memory
        ::size_t numEdges{0};
        PackedDNASequence baseValues{};
        ::size_t* columnValues{nullptr};
        ::size_t* rowOffsets{nullptr};
        bool ownsInstance{false};
        PangenomeGraph* pinned_instance{nullptr};
        PangenomeGraph* d_instance{nullptr};

        // Pangenome graph constructor
        __host__ __device__ __forceinline__ PangenomeGraph(const ::size_t numEdges, const PackedDNASequence& baseValues, ::size_t* const columnValues, ::size_t* const rowOffsets, const bool ownsInstance, PangenomeGraph* const pinned_instance = nullptr, PangenomeGraph* const d_instance = nullptr) : numEdges(numEdges), baseValues(baseValues), columnValues(columnValues), rowOffsets(rowOffsets), ownsInstance(ownsInstance), pinned_instance(pinned_instance), d_instance(d_instance) {}
    };
} // cuSGA

#endif //CUSGA_PANGENOMEGRAPH_CUH
