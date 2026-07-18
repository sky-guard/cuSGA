#ifndef CUSGA_SEQUENCEGRAPH_CUH
#define CUSGA_SEQUENCEGRAPH_CUH
#include <cuda/atomic>

#include "DoubleBuffer.cuh"
#include "Frontier.cuh"
#include "PackedDNASequence.cuh"
#include "PangenomeGraph.cuh"

namespace cuSGA {
    // Forward declaration
    class SequenceGraph;

    // Sequence graph kernels
    namespace SequenceGraphKernels {
        // Initialization kernel
        __global__ void initialize(SequenceGraph d_sequenceGraph);

        // Substitutions kernel
        __global__ void substitutions(SequenceGraph d_sequenceGraph, ::size_t layerIdx);

        // Deletions kernel
        __global__ void deletions(SequenceGraph d_sequenceGraph);

        // Insertions kernel
        __global__ void insertions(SequenceGraph d_sequenceGraph, Frontier d_frontier);

        // Propagations kernel
        __global__ void propagations(SequenceGraph d_sequenceGraph, Frontier d_frontier, ::size_t layerIdx);

        // Minimum score kernel
        inline constexpr unsigned MIN_SCORE_SHUFFLE_MASK{0xFFFFFFFF};
        __host__ __device__ __forceinline__ inline ::size_t minScoreSMemCalculator(const ::size_t blockSize) {
            // Get number of warps in the block
            const auto numWarps{blockSize / KernelUtils::WARP_SIZE};

            return numWarps * sizeof(::size_t);
        }
        __global__ void minScore(SequenceGraph d_sequenceGraph, ::size_t scoreIdx);
    } // SequenceGraphKernels

    // Sequence graph
    class SequenceGraph {
    public:
        // Alignment related constants
        static constexpr ::uint8_t INITIALIZATION_COST{1};
        static constexpr ::uint8_t SUBSTITUTION_COST{1};
        static constexpr ::uint8_t DELETION_COST{1};
        static constexpr ::uint8_t INSERTION_COST{1};
        static constexpr ::size_t SCORE_MAX_VALUE{::std::numeric_limits<::uint64_t>::max()};

        // Grow allocator using the expected buffers size
        __host__ __device__ __forceinline__ static void growBuffers(KernelUtils::BumpPtrAllocator* allocator, const ::size_t maxSequenceLength, const ::size_t numNodes, const ::size_t numEdges, const ::size_t totalNumConnectedComponents, const ::size_t numScores) {
            // Grow size for sequence
            PackedDNASequence::growBuffers(allocator, maxSequenceLength);

            // Grow size for pangenome graph
            PangenomeGraph::growBuffers(allocator, numNodes, numEdges);

            // Grow size for connected components offsets
            allocator->grow<::std::remove_pointer_t<decltype(connectedComponentsOffsets)>>(totalNumConnectedComponents);

            // Grow size for connected components mappings
            allocator->grow<::std::remove_pointer_t<decltype(connectedComponentsMappings)>>(NUM_BASES * numNodes);

            // Grow size for costs double buffer
            DoubleBuffer<::size_t>::growBuffers(allocator, numNodes);

            // Grow size for scores
            allocator->grow<::std::remove_pointer_t<decltype(scores)>>(numScores);
        }

        // Default constructor
        SequenceGraph() = default;
        // Parameterized constructor
        __host__ SequenceGraph(const ::std::string& pangenomeGraphFileName, ::std::string const (& connectedComponentsFileNames)[NUM_BASES], bool ownsInstance, SequenceGraph* pinned_instanceOptional = nullptr, SequenceGraph* d_instanceOptional = nullptr);
        // Copy constructor
        SequenceGraph(const SequenceGraph& other) = default;
        // Move constructor
        SequenceGraph(SequenceGraph&& other) = default;
        // Copy assignment
        SequenceGraph& operator=(const SequenceGraph& other) = default;
        // Move assignment
        SequenceGraph& operator=(SequenceGraph&& other) = default;
        // Destructor
        ~SequenceGraph() = default;

        // Move sequence graph to device
        __host__ SequenceGraph* copyToDevice(SequenceGraph* d_instanceOptional = nullptr, KernelUtils::BumpPtrAllocator* allocatorOptional = nullptr);
        // Free sequence graph
        __host__ void free() const;

        // Get sequence
        __host__ __device__ __forceinline__ const PackedDNASequence& getSequence() const {
            return sequence;
        }

        // Get pangenome graph
        __host__ __device__ __forceinline__ const PangenomeGraph& getPangenomeGraph() const {
            return pangenomeGraph;
        }

        // Get number of connected components for a given character graph DNA base
        __host__ __device__ __forceinline__ ::size_t getNumConnectedComponents(const DNABase characterGraphBase) const {
            return numConnectedComponents[static_cast<::size_t>(characterGraphBase)];
        }

        // Get connected component offset for a given character graph DNA base and connected component index
        __host__ __device__ __forceinline__ ::size_t getConnectedComponentOffset(const DNABase characterGraphBase, const ::size_t connectedComponentIdx) const {
            return connectedComponentsOffsets[static_cast<::size_t>(characterGraphBase)][connectedComponentIdx];
        }

        // Get connected component mapping for a given character graph DNA base and node index
        __host__ __device__ __forceinline__ ::size_t getConnectedComponentMapping(DNABase characterGraphBase, const ::size_t nodeIdx) const {
            return connectedComponentsMappings[static_cast<::size_t>(characterGraphBase)][nodeIdx];
        }

        // Get costs double buffer
        __host__ __device__ __forceinline__ const DoubleBuffer<::cuda::atomic<::uint64_t, ::cuda::thread_scope_device>>& getCostsDoubleBuffer() const {
            return costsDoubleBuffer;
        }

        // Get scores
        __host__ __device__ __forceinline__ ::cuda::atomic<::uint64_t, ::cuda::thread_scope_device>* getScores() const {
            return scores;
        }

        // Copy back scores from device
        __host__ __forceinline__ ::cuda::atomic<::uint64_t, ::cuda::thread_scope_device>* h2d_getScores() const {
            if (d_instance) {
                CUDA_CHECK(::cudaMemcpy(scores, pinned_instance->scores, numScores * sizeof(scores[0]),::cudaMemcpyDeviceToHost));
            }

            return scores;
        }

        // Get CUDA atomic score for a given index
        __host__ __device__ __forceinline__ ::cuda::atomic<::uint64_t, ::cuda::thread_scope_device>& getAtomicScore(const ::size_t idx) const {
            return scores[idx];
        }

        // Get pinned instance
        __host__ __device__ __forceinline__ SequenceGraph* getPinnedInstance() const {
            return pinned_instance;
        }

        // Get device instance
        __host__ __device__ __forceinline__ SequenceGraph* getDeviceInstance() const {
            return d_instance;
        }

        // Get buffer root
        __host__ __device__ __forceinline__ void* getBuffersRoot() const {
            return sequence.getBuffersRoot();
        }

        // Reset scores
        __host__ __device__ __forceinline__ void initializeScores() const {
            ::memset(scores, 1, numScores * sizeof(scores[0]));
        }

        // Reset device scores
        __host__ __forceinline__ void h2d_initializeScores() const {
            if (d_instance) {
                CUDA_CHECK(::cudaMemsetAsync(pinned_instance->scores, 1, numScores * sizeof(scores[0]), cudaStreamDefault));
            }
        }

        // Align sequence
        __host__ void align(const ::std::string& sequenceFileName);

        // Launch initialize kernel
        __host__ __forceinline__ void initialize() const {
            // Get block size
            const auto blockSize{KernelUtils::cudaSizeBlock<SequenceGraphKernels::initialize>()};

            // Get grid size
            const auto gridSize{KernelUtils::cudaSizeGrid(pangenomeGraph.getNumNodes(), blockSize)};

            // Launch kernel
            KernelUtils::cudaLaunchKernel<SequenceGraphKernels::initialize>(gridSize, blockSize, 0, cudaStreamDefault, *pinned_instance);
        }

        // Launch substitutions kernel
        __host__ __forceinline__ void substitutions(const ::size_t layerIdx) const {
            // Get block size
            const auto blockSize{KernelUtils::cudaSizeBlock<SequenceGraphKernels::substitutions>()};

            // Get grid size
            const auto gridSize{KernelUtils::cudaSizeGrid(pangenomeGraph.getNumNodes(), blockSize)};

            // Launch kernel
            KernelUtils::cudaLaunchKernel<SequenceGraphKernels::substitutions>(gridSize, blockSize, 0, cudaStreamDefault, *pinned_instance, layerIdx);
        }

        // Launch deletions kernel
        __host__ __forceinline__ void deletions(const ::size_t layerIdx) const {
            // Get block size
            const auto blockSize{KernelUtils::cudaSizeBlock<SequenceGraphKernels::deletions>()};

            // Get grid size
            const auto gridSize{KernelUtils::cudaSizeGrid(pangenomeGraph.getNumNodes(), blockSize)};

            // Launch kernel
            KernelUtils::cudaLaunchKernel<SequenceGraphKernels::deletions>(gridSize, blockSize, 0, cudaStreamDefault, *pinned_instance, layerIdx);
        }

        // Launch insertions kernel
        __host__ __forceinline__ void insertions(const Frontier& d_frontier) const {
            // Get block size
            const auto blockSize{KernelUtils::cudaSizeBlock<SequenceGraphKernels::insertions>()};

            // Get grid size
            const auto gridSize{KernelUtils::cudaSizeGrid(pangenomeGraph.getNumNodes(), blockSize)};

            // Launch kernel
            KernelUtils::cudaLaunchKernel<SequenceGraphKernels::insertions>(gridSize, blockSize, 0, cudaStreamDefault, *pinned_instance, d_frontier);
        }

        // Launch propagations kernel
        __host__ __forceinline__ void propagations(const Frontier& d_frontier, const ::size_t layerIdx) const {
            // Get block size
            const auto blockSize{KernelUtils::cudaSizeBlock<SequenceGraphKernels::propagations>()};

            // Get grid size
            const auto gridSize{KernelUtils::cudaSizeGrid(d_frontier.getSize(), blockSize)};

            // Launch kernel
            KernelUtils::cudaLaunchKernel<SequenceGraphKernels::propagations>(gridSize, blockSize, 0, cudaStreamDefault, *pinned_instance, d_frontier, layerIdx);
        }

        // Launch min score kernel
        __host__ __forceinline__ void minScore(const ::size_t scoreIdx) const {
            // Get block size and round it down to be a multiple of WARP_SIZE
            auto blockSize{KernelUtils::cudaSizeBlockWithDynamicSMem<SequenceGraphKernels::minScore, SequenceGraphKernels::minScoreSMemCalculator>()};
            blockSize &= ~(KernelUtils::WARP_SIZE - 1);

            // Get grid size
            const auto gridSize{KernelUtils::cudaSizeGrid(pangenomeGraph.getNumNodes(), blockSize)};

            // Get dynamic shared memory size
            const auto dynamicSMemSize{SequenceGraphKernels::minScoreSMemCalculator(blockSize)};

            // Launch kernel
            KernelUtils::cudaLaunchKernel<SequenceGraphKernels::minScore>(gridSize, blockSize, dynamicSMemSize, cudaStreamDefault, *pinned_instance, scoreIdx);
        }

    private:
        // Sequence graph implementation
        PackedDNASequence sequence{};
        PangenomeGraph pangenomeGraph{};
        ::size_t numConnectedComponents[NUM_BASES]{};
        ::size_t* connectedComponentsOffsets[NUM_BASES]{nullptr};
        ::size_t* connectedComponentsMappings[NUM_BASES]{nullptr};
        DoubleBuffer<::cuda::atomic<::uint64_t, ::cuda::thread_scope_device>> costsDoubleBuffer{};
        ::size_t numScores{0};
        ::cuda::atomic<::uint64_t, ::cuda::thread_scope_device>* scores{nullptr};
        bool ownsInstance{false};
        SequenceGraph* pinned_instance{nullptr};
        SequenceGraph* d_instance{nullptr};

        // Sequence graph constructor
        __host__ __device__ __forceinline__ SequenceGraph(const PackedDNASequence& sequence, const PangenomeGraph& pangenomeGraph, ::size_t const (& numConnectedComponents)[NUM_BASES], ::size_t* const (& connectedComponentsOffsets)[NUM_BASES], ::size_t* const (& connectedComponentsMappings)[NUM_BASES], const DoubleBuffer<::cuda::atomic<::uint64_t, ::cuda::thread_scope_device>>& costsDoubleBuffer, const ::size_t numScores, ::cuda::atomic<::uint64_t, ::cuda::thread_scope_device>* const scores, const bool ownsInstance, SequenceGraph* const pinned_instance = nullptr, SequenceGraph* const d_instance = nullptr) : sequence(sequence), pangenomeGraph(pangenomeGraph), costsDoubleBuffer(costsDoubleBuffer), numScores(numScores), scores(scores), ownsInstance(ownsInstance), pinned_instance(pinned_instance), d_instance(d_instance) {
#pragma unroll
            for (::size_t i{0}; i < NUM_BASES; ++i) {
                // Set number of cuts
                this->numConnectedComponents[i] = numConnectedComponents[i];

                // Set cuts buffers
                this->connectedComponentsOffsets[i] = connectedComponentsOffsets[i];

                // Set components buffers
                this->connectedComponentsMappings[i] = connectedComponentsMappings[i];
            }
        }
    };
} // cuSGA

#endif //CUSGA_SEQUENCEGRAPH_CUH
