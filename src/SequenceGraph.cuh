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

        // Create sequence graph from file
        __host__ static SequenceGraph* createFromFiles(const ::std::string& pangenomeGraphFileName);

        // Move sequence graph to device
        __host__ SequenceGraph* copyToDevice();
        // Free sequence graph
        __host__ void free() const;

        // Get sequence
        __host__ __device__ PackedDNASequence* getSequence() const;
        // Get pangenome graph
        __host__ __device__ PangenomeGraph* getPangenomeGraph() const;
        // Get costs double buffer
        __host__ __device__ DoubleBuffer<::cuda::atomic<::uint64_t, ::cuda::thread_scope_device>>* getCostsDoubleBuffer() const;
        // Get score
        __host__ __device__ ::uint64_t getScore() const;
        // Copy back score from device
        __host__ ::uint64_t getScoreSync();
        // Get pointer to CUDA atomic score
        __host__ __device__ ::cuda::atomic<::uint64_t, ::cuda::thread_scope_device>& getAtomicScore();
        // Get device instance
        __host__ __device__ SequenceGraph* getDeviceInstance() const;

        // Reset score
        __host__ __device__ void resetScore();
        // Reset device score
        __host__ void resetScoreSync() const;

        // Align sequence
        __host__ ::std::vector<::uint64_t> align(const ::std::string& sequenceFileName);
        // Perform initialization step
        __host__ void initialize(bool sync = true);
        // Perform substitutions for a given layer index
        __host__ void substitutions(::size_t layerIdx, bool sync = true) const;
        // Perform deletions for a given layer index
        __host__ void deletions(::size_t layerIdx, bool sync = true) const;
        // Perform insertions for a given layer index
        __host__ void insertions(const Frontier* frontier, bool sync = true) const;
        // Perform propagations for a given layer index
        __host__ void propagations(::size_t layerIdx, const Frontier* frontier, bool sync = true) const;
        // Compute minimum score
        __host__ void minScore(bool sync = true) const;

    private:
        // Sequence graph implementation
        PackedDNASequence* sequence{nullptr};
        PangenomeGraph* pangenomeGraph{nullptr};
        DoubleBuffer<::cuda::atomic<::uint64_t, ::cuda::thread_scope_device>>* costsDoubleBuffer{nullptr};
        ::cuda::atomic<::uint64_t, ::cuda::thread_scope_device> score{SCORE_MAX_VALUE};
        SequenceGraph* d_instance{nullptr};

        // Default constructor
        SequenceGraph() = default;
        // Sequence graph constructor
        __host__ __device__ SequenceGraph(PackedDNASequence* sequence, PangenomeGraph* pangenomeGraph, DoubleBuffer<::cuda::atomic<::uint64_t, ::cuda::thread_scope_device>>* costsDoubleBuffer = nullptr, ::uint64_t score = SCORE_MAX_VALUE, SequenceGraph* d_instance = nullptr);
    };

    namespace SequenceGraphKernels {
        // Initialization kernel
        inline constexpr ::size_t INITIALIZE_DYNAMIC_SMEM_SIZE{0};
        inline constexpr int INITIALIZE_MAX_BLOCK_SIZE{0};
        __global__ void initialize(const SequenceGraph* sequenceGraph);

        // Substitutions kernel
        inline constexpr ::size_t SUBSTITUTIONS_DYNAMIC_SMEM_SIZE{0};
        inline constexpr int SUBSTITUTIONS_MAX_BLOCK_SIZE{0};
        __global__ void substitutions(const SequenceGraph* sequenceGraph, ::size_t layerIdx);

        // Deletions kernel
        inline constexpr ::size_t DELETIONS_DYNAMIC_SMEM_SIZE{0};
        inline constexpr int DELETIONS_MAX_BLOCK_SIZE{0};
        __global__ void deletions(const SequenceGraph* sequenceGraph, ::size_t layerIdx);

        // Insertions kernel
        inline constexpr ::size_t INSERTIONS_DYNAMIC_SMEM_SIZE{0};
        inline constexpr int INSERTIONS_MAX_BLOCK_SIZE{0};
        __global__ void insertions(const SequenceGraph* sequenceGraph, Frontier* frontier);

        // Propagations kernel
        inline constexpr ::size_t PROPAGATIONS_DYNAMIC_SMEM_SIZE{0};
        inline constexpr int PROPAGATIONS_MAX_BLOCK_SIZE{0};
        __global__ void propagations(const SequenceGraph* sequenceGraph, Frontier* frontier, ::size_t layerIdx);

        // Minimum score kernel
        inline constexpr ::size_t MIN_SCORE_DYNAMIC_SMEM_SIZE{0};
        inline constexpr int MIN_SCORE_MAX_BLOCK_SIZE{0};
        inline constexpr unsigned MIN_SCORE_SHUFFLE_MASK{0xFFFFFFFF};
        __global__ void minScore(SequenceGraph* sequenceGraph);
    } // SequenceGraphKernels
} // cuSGA

#endif //CUSGA_SEQUENCEGRAPH_CUH
