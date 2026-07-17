#ifndef CUSGA_SEQUENCEGRAPH_CUH
#define CUSGA_SEQUENCEGRAPH_CUH
#include <cuda/atomic>

#include "DoubleBuffer.cuh"
#include "Frontier.cuh"
#include "PackedDNASequence.cuh"
#include "PangenomeGraph.cuh"

namespace cuSGA {
    class SequenceGraph {
    public:
        // Alignment related constants
        static constexpr ::uint8_t INITIALIZATION_COST{1};
        static constexpr ::uint8_t SUBSTITUTION_COST{1};
        static constexpr ::uint8_t DELETION_COST{1};
        static constexpr ::uint8_t INSERTION_COST{1};
        static constexpr ::size_t SCORE_MAX_VALUE{::std::numeric_limits<::uint64_t>::max()};

        // Grow allocator using the expected buffers size
        __host__ __device__ __forceinline__ static void growBuffers(KernelUtils::BumpPtrAllocator* allocator) {
            // TODO: decide input parameters
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
        __host__ __device__ __forceinline__ ::cuda::atomic<::uint64_t, ::cuda::thread_scope_device>* d_getScores() const {
            if (d_instance) {
                KernelUtils::cudaMemcpyAsync(scores, pinned_instance->scores, numScores * sizeof(scores[0]),::cudaMemcpyDeviceToHost, cudaStreamDefault);
            }

            return scores;
        }

        // Get CUDA atomic score for a given index
        __host__ __device__ __forceinline__ const ::cuda::atomic<::uint64_t, ::cuda::thread_scope_device>& getAtomicScore(const ::size_t idx) const {
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

        // Reset scores
        __host__ __device__ __forceinline__ void resetScores() const {
            ::memset(scores, 1, numScores * sizeof(scores[0]));
        }

        // Reset device scores
        __host__ __forceinline__ void d_resetScores() const {
            if (d_instance) {
                KernelUtils::cudaMemsetAsync(pinned_instance->scores, 1, numScores * sizeof(scores[0]), cudaStreamDefault);
            }
        }

        // TODO: check and possibly force inline
        // Align sequence
        __host__ ::std::vector<::uint64_t> align(const ::std::string& sequenceFileName);
        // Perform initialization step
        __host__ void initialize(bool sync = true) const;
        // Perform substitutions for a given layer index
        __host__ void substitutions(::size_t layerIdx, bool sync = true) const;
        // Perform deletions for a given layer index
        __host__ void deletions(::size_t layerIdx, bool sync = true) const;
        // Perform insertions for a given layer index
        __host__ void insertions(const Frontier& d_frontier, bool sync = true) const;
        // Perform propagations for a given layer index
        __host__ void propagations(const Frontier& d_frontier, ::size_t layerIdx, bool sync = true) const;
        // Compute minimum score
        __host__ void minScore(bool sync = true) const;

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

    namespace SequenceGraphKernels {
        // Initialization kernel
        inline constexpr ::size_t INITIALIZE_DYNAMIC_SMEM_SIZE{0};
        inline constexpr int INITIALIZE_MAX_BLOCK_SIZE{0};
        __global__ void initialize(SequenceGraph d_sequenceGraph);

        // Substitutions kernel
        inline constexpr ::size_t SUBSTITUTIONS_DYNAMIC_SMEM_SIZE{0};
        inline constexpr int SUBSTITUTIONS_MAX_BLOCK_SIZE{0};
        __global__ void substitutions(SequenceGraph d_sequenceGraph, ::size_t layerIdx);

        // Deletions kernel
        inline constexpr ::size_t DELETIONS_DYNAMIC_SMEM_SIZE{0};
        inline constexpr int DELETIONS_MAX_BLOCK_SIZE{0};
        __global__ void deletions(SequenceGraph d_sequenceGraph);

        // Insertions kernel
        inline constexpr ::size_t INSERTIONS_DYNAMIC_SMEM_SIZE{0};
        inline constexpr int INSERTIONS_MAX_BLOCK_SIZE{0};
        __global__ void insertions(SequenceGraph d_sequenceGraph, Frontier d_frontier);

        // Propagations kernel
        inline constexpr ::size_t PROPAGATIONS_DYNAMIC_SMEM_SIZE{0};
        inline constexpr int PROPAGATIONS_MAX_BLOCK_SIZE{0};
        __global__ void propagations(SequenceGraph d_sequenceGraph, Frontier d_frontier, ::size_t layerIdx);

        // Minimum score kernel
        inline constexpr ::size_t MIN_SCORE_DYNAMIC_SMEM_SIZE{0};
        inline constexpr int MIN_SCORE_MAX_BLOCK_SIZE{0};
        inline constexpr unsigned MIN_SCORE_SHUFFLE_MASK{0xFFFFFFFF};
        __global__ void minScore(SequenceGraph d_sequenceGraph);
    } // SequenceGraphKernels
} // cuSGA

#endif //CUSGA_SEQUENCEGRAPH_CUH
