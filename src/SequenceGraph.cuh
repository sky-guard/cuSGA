#ifndef CUSGA_SEQUENCEGRAPH_CUH
#define CUSGA_SEQUENCEGRAPH_CUH
#include "DoubleBuffer.cuh"
#include "Frontier.cuh"
#include "PackedDNASequence.cuh"
#include "PangenomeGraph.cuh"

namespace cuSGA {
    // Define connected component size type
    using connectedComponentSize_t = targetSize_t;

    // Define cost type
    using cost_t = targetSize_t;

    // Define score size type
    using scoreSize_t = targetSize_t;

    // Sequence graph forward declaration
    class SequenceGraph;

    // Sequence graph kernels
    namespace SequenceGraphKernels {
        // Initialization kernel
        __global__ void initialize(const sequencePack_t* __restrict__ d_baseValues, cost_t* __restrict__ d_currentCosts, nodeSize_t numNodes, DNABase initialSequenceBase);

        // Substitutions kernel
        __global__ void substitutions(const edgeSize_t* __restrict__ d_neighborOffsets, const nodeSize_t* __restrict__ d_neighborValues, const sequencePack_t* __restrict__ d_baseValues, const cost_t* __restrict__ d_previousCosts, cost_t* __restrict__ d_currentCosts, bool* __restrict__ d_needsVisiting, const connectedComponentSize_t* __restrict__ connectedComponentsReverseMapping, nodeSize_t numNodes, DNABase sequenceBase);

        // Deletions kernel
        __global__ void deletions(const cost_t* __restrict__ d_previousCosts, cost_t* __restrict__ d_currentCosts, nodeSize_t numNodes);

        // Insertions and propagations kernel
        inline constexpr targetSize_t SHARED_FRONTIER_BUFFER_SIZE{KernelUtils::WARP_SIZE << 1};
        __device__ __forceinline__ void syncFrontierSizes(Frontier* __restrict__ shared_frontier, Frontier* __restrict__ warpFrontier, unsigned mask);
        __device__ __forceinline__ void processNeighbor(Frontier* __restrict__ warpFrontier, Frontier* __restrict__ shared_frontier, cost_t* __restrict__ d_currentCosts, nodeSize_t neighborIdx, nodeSize_t neighborLocalIdx, cost_t updatedCurrentLayerNeighborCost);
        __device__ __forceinline__ void processNode(nodeSize_t nodeIdx, Frontier* __restrict__ warpFrontier, Frontier* __restrict__ shared_frontier, const edgeSize_t* __restrict__ d_neighborOffsets, const nodeSize_t* __restrict__ d_neighborValues, const connectedComponentSize_t* __restrict__ d_connectedComponentLocalIndexMappings, cost_t* __restrict__ d_currentCosts);
        __global__ void insertionsAndPropagations(const edgeSize_t* __restrict__ d_neighborOffsets, const nodeSize_t* __restrict__ d_neighborValues, const nodeSize_t* __restrict__ d_connectedComponentOffsets, const nodeSize_t* __restrict__ d_connectedComponentMappings, const connectedComponentSize_t* __restrict__ d_connectedComponentLocalIndexMappings, cost_t* __restrict__ d_currentCosts, nodeSize_t* __restrict__ d_buffers, bool* __restrict__ d_needsVisiting, targetSize_t numWarpsPerBlock, connectedComponentSize_t numConnectedComponents, connectedComponentSize_t maxConnectedComponentSize);

        // Minimum cost kernel
        __global__ void minCost(const cost_t* __restrict__ d_currentCosts, cost_t* __restrict__ d_scores, nodeSize_t numNodes, scoreSize_t scoreIdx);
    } // SequenceGraphKernels

    // Sequence graph
    class SequenceGraph {
    public:
        // Alignment related constants
        static constexpr ::uint8_t INITIALIZATION_COST{1};
        static constexpr ::uint8_t SUBSTITUTION_COST{1};
        static constexpr ::uint8_t DELETION_COST{1};
        static constexpr ::uint8_t INSERTION_COST{1};
        static constexpr cost_t COST_MAX_VALUE{::std::numeric_limits<cost_t>::max()};

        // Grow allocator using the expected buffers size
        __host__ __device__ __forceinline__ static void growBuffers(KernelUtils::BumpPtrAllocator* allocator, const nodeSize_t numNodes, const edgeSize_t numEdges, const sequenceSize_t maxSequenceLength, const connectedComponentSize_t totalNumConnectedComponents, const scoreSize_t numScores) {
            // Grow size for pangenome graph
            PangenomeGraph::growBuffers(allocator, numNodes, numEdges);

            // Grow size for sequence
            PackedDNASequence::growBuffers(allocator, maxSequenceLength);

            // Grow size for connected components offsets
            allocator->grow<::std::remove_reference_t<decltype(connectedComponentsOffsets[0])>>(totalNumConnectedComponents + NUM_BASES);

            // Grow size for connected components mappings
            allocator->grow<::std::remove_reference_t<decltype(connectedComponentsMappings[0])>>(NUM_BASES * numNodes);

            // Grow size for connected components reverse mappings
            allocator->grow<::std::remove_reference_t<decltype(connectedComponentsReverseMappings[0])>>(NUM_BASES * numNodes);

            // Grow size for connected components local index mappings
            allocator->grow<::std::remove_reference_t<decltype(connectedComponentsLocalIndexMappings[0])>>(NUM_BASES * numNodes);

            // Grow size for costs double buffer
            DoubleBuffer<cost_t>::growBuffers(allocator, numNodes);

            // Grow size for scores
            allocator->grow<::std::remove_reference_t<decltype(scores[0])>>(numScores);
        }

        // Grow allocator using the expected buffers size, excluding the sequence
        // NOTE: There is no need to copy the sequence over to the device!
        __host__ __device__ __forceinline__ static void growBuffersWithoutSequence(KernelUtils::BumpPtrAllocator* allocator, const nodeSize_t numNodes, const edgeSize_t numEdges, const connectedComponentSize_t totalNumConnectedComponents, const scoreSize_t numScores) {
            // Grow size for pangenome graph
            PangenomeGraph::growBuffers(allocator, numNodes, numEdges);

            // Grow size for connected components offsets
            allocator->grow<::std::remove_reference_t<decltype(connectedComponentsOffsets[0])>>(totalNumConnectedComponents + NUM_BASES);

            // Grow size for connected components mappings
            allocator->grow<::std::remove_reference_t<decltype(connectedComponentsMappings[0])>>(NUM_BASES * numNodes);

            // Grow size for connected components reverse mappings
            allocator->grow<::std::remove_reference_t<decltype(connectedComponentsReverseMappings[0])>>(NUM_BASES * numNodes);

            // Grow size for connected components local index mappings
            allocator->grow<::std::remove_reference_t<decltype(connectedComponentsLocalIndexMappings[0])>>(NUM_BASES * numNodes);

            // Grow size for costs double buffer
            DoubleBuffer<cost_t>::growBuffers(allocator, numNodes);

            // Grow size for scores
            allocator->grow<::std::remove_reference_t<decltype(scores[0])>>(numScores);
        }

        // Default constructor
        SequenceGraph() = default;
        // Parameterized constructor
        __host__ SequenceGraph(const ::std::string& pangenomeGraphFileName, const ::std::string& sequenceFileName, ::std::string const (& connectedComponentsFileNames)[NUM_BASES], bool ownsInstance, SequenceGraph* __restrict__ pinned_instanceOptional = nullptr, KernelUtils::BumpPtrAllocator* allocatorOptional = nullptr);

        // Sequence graph constructor
        __host__ __device__ __forceinline__ SequenceGraph(const PangenomeGraph& __restrict__ pangenomeGraph, const PackedDNASequence& __restrict__ sequence, connectedComponentSize_t const (& numConnectedComponents)[NUM_BASES], nodeSize_t* __restrict__ const (& connectedComponentsOffsets)[NUM_BASES], nodeSize_t* __restrict__ const (& connectedComponentsMappings)[NUM_BASES], connectedComponentSize_t* __restrict__ const (& connectedComponentsReverseMappings)[NUM_BASES], connectedComponentSize_t* __restrict__ const (& connectedComponentsLocalIndexMappings)[NUM_BASES], connectedComponentSize_t const (& maxConnectedComponentsSizes)[NUM_BASES], const DoubleBuffer<cost_t>& __restrict__ costsDoubleBuffer, const scoreSize_t numScores, cost_t* __restrict__ const scores, const bool ownsInstance, SequenceGraph* __restrict__ const pinned_instance = nullptr, SequenceGraph* __restrict__ const d_instance = nullptr) : pangenomeGraph{pangenomeGraph}, sequence{sequence}, costsDoubleBuffer{costsDoubleBuffer}, numScores{numScores}, scores{scores}, ownsInstance{ownsInstance}, pinned_instance{pinned_instance}, d_instance{d_instance} {
#pragma unroll
            for (DNABase_t baseIdx{0}; baseIdx < NUM_BASES; ++baseIdx) {
                // Set number of connected components
                this->numConnectedComponents[baseIdx] = numConnectedComponents[baseIdx];

                // Set connected components offsets
                this->connectedComponentsOffsets[baseIdx] = connectedComponentsOffsets[baseIdx];

                // Set connected components mappings
                this->connectedComponentsMappings[baseIdx] = connectedComponentsMappings[baseIdx];

                // Set connected components reverse mappings
                this->connectedComponentsReverseMappings[baseIdx] = connectedComponentsReverseMappings[baseIdx];

                // Set connected component local index mappings
                this->connectedComponentsLocalIndexMappings[baseIdx] = connectedComponentsLocalIndexMappings[baseIdx];

                // Set max connected components sizes
                this->maxConnectedComponentsSizes[baseIdx] = maxConnectedComponentsSizes[baseIdx];
            }
        }

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
        __host__ SequenceGraph copyToDevice(SequenceGraph* __restrict__ d_instanceOptional = nullptr, KernelUtils::BumpPtrAllocator* allocatorOptional = nullptr);
        // Free sequence graph
        __host__ void free() const;

        // Get pangenome graph
        __host__ __device__ __forceinline__ const PangenomeGraph& __restrict__ getPangenomeGraph() const {
            return pangenomeGraph;
        }

        // Get sequence
        __host__ __device__ __forceinline__ const PackedDNASequence& __restrict__ getSequence() const {
            return sequence;
        }

        // Get number of connected components for a given character graph DNA base
        __host__ __device__ __forceinline__ connectedComponentSize_t getNumConnectedComponents(const DNABase characterGraphBase) const {
            return numConnectedComponents[static_cast<DNABase_t>(characterGraphBase)];
        }

        // Get connected component offset for a given character graph DNA base and connected component index
        __host__ __device__ __forceinline__ nodeSize_t getConnectedComponentOffset(const DNABase characterGraphBase, const connectedComponentSize_t connectedComponentIdx) const {
            return connectedComponentsOffsets[static_cast<DNABase_t>(characterGraphBase)][connectedComponentIdx];
        }

        // Get connected component mapping for a given character graph DNA base and node index
        __host__ __device__ __forceinline__ nodeSize_t getConnectedComponentMapping(const DNABase characterGraphBase, const nodeSize_t nodeIdx) const {
            return connectedComponentsMappings[static_cast<DNABase_t>(characterGraphBase)][nodeIdx];
        }

        // Get connected component reverse mappings for a given character graph DNA base
        __host__ __device__ __forceinline__ connectedComponentSize_t* __restrict__ getConnectedComponentReverseMappings(const DNABase characterGraphBase) const {
            return connectedComponentsReverseMappings[static_cast<DNABase_t>(characterGraphBase)];
        }

        // Get connected component reverse mapping for a given character graph DNA base and node index
        __host__ __device__ __forceinline__ connectedComponentSize_t getConnectedComponentReverseMapping(const DNABase characterGraphBase, const nodeSize_t nodeIdx) const {
            return connectedComponentsReverseMappings[static_cast<DNABase_t>(characterGraphBase)][nodeIdx];
        }

        // Get max connected component size for a given character graph DNA base
        __host__ __device__ __forceinline__ connectedComponentSize_t getMaxConnectedComponentSize(const DNABase characterGraphBase) const {
            return maxConnectedComponentsSizes[static_cast<DNABase_t>(characterGraphBase)];
        }

        // Get connected component local index mapping for a given character graph DNA base and node index
        __host__ __device__ __forceinline__ connectedComponentSize_t getConnectedComponentLocalIndexMapping(const DNABase characterGraphBase, const nodeSize_t nodeIdx) const {
            return connectedComponentsLocalIndexMappings[static_cast<DNABase_t>(characterGraphBase)][nodeIdx];
        }

        // Get costs double buffer
        __host__ __device__ __forceinline__ const DoubleBuffer<cost_t>& __restrict__ getCostsDoubleBuffer() const {
            return costsDoubleBuffer;
        }

        // Get number of scores
        __host__ __device__ __forceinline__ scoreSize_t getNumScores() const {
            return numScores;
        }

        // Get scores
        __host__ __device__ __forceinline__ cost_t* __restrict__ getScores() const {
            return scores;
        }

        // Copy back scores from device
        __host__ __forceinline__ cost_t* __restrict__ h2d_getScores() const {
            if (d_instance) {
                CUDA_CHECK(::cudaMemcpy(scores, pinned_instance->scores, numScores * sizeof(scores[0]),::cudaMemcpyDeviceToHost));
            }

            return scores;
        }

        // Get pinned instance
        __host__ __device__ __forceinline__ SequenceGraph* __restrict__ getPinnedInstance() const {
            return pinned_instance;
        }

        // Get device instance
        __host__ __device__ __forceinline__ SequenceGraph* __restrict__ getDeviceInstance() const {
            return d_instance;
        }

        // Get buffer root
        __host__ __device__ __forceinline__ void* __restrict__ getBuffersRoot() const {
            return pangenomeGraph.getBuffersRoot();
        }

        // Shuffle object from the given lane, with the given mask
        __device__ __forceinline__ void shuffle_sync(const unsigned mask, const int srcLaneIdx) {
            this->pangenomeGraph.shuffle_sync(mask, srcLaneIdx);
            this->sequence.shuffle_sync(mask, srcLaneIdx);
#pragma unroll
            for (DNABase_t baseIdx{0}; baseIdx < NUM_BASES; ++baseIdx) {
                this->numConnectedComponents[baseIdx] = ::__shfl_sync(mask, numConnectedComponents[baseIdx], srcLaneIdx);
                this->connectedComponentsOffsets[baseIdx] = reinterpret_cast<::std::remove_reference_t<decltype(connectedComponentsOffsets[0])>>(::__shfl_sync(mask, reinterpret_cast<unsigned long long>(connectedComponentsOffsets[baseIdx]), srcLaneIdx));
                this->connectedComponentsMappings[baseIdx] = reinterpret_cast<::std::remove_reference_t<decltype(connectedComponentsMappings[0])>>(::__shfl_sync(mask, reinterpret_cast<unsigned long long>(connectedComponentsMappings[baseIdx]), srcLaneIdx));
                this->connectedComponentsReverseMappings[baseIdx] = reinterpret_cast<::std::remove_reference_t<decltype(connectedComponentsReverseMappings[0])>>(::__shfl_sync(mask, reinterpret_cast<unsigned long long>(connectedComponentsReverseMappings[baseIdx]), srcLaneIdx));
                this->connectedComponentsLocalIndexMappings[baseIdx] = reinterpret_cast<::std::remove_reference_t<decltype(connectedComponentsLocalIndexMappings[0])>>(::__shfl_sync(mask, reinterpret_cast<unsigned long long>(connectedComponentsLocalIndexMappings[baseIdx]), srcLaneIdx));
                this->maxConnectedComponentsSizes[baseIdx] = ::__shfl_sync(mask, maxConnectedComponentsSizes[baseIdx], srcLaneIdx);
            }
            this->costsDoubleBuffer.shuffle_sync(mask, srcLaneIdx);
            this->numScores = ::__shfl_sync(mask, numScores, srcLaneIdx);
            this->scores = reinterpret_cast<decltype(scores)>(::__shfl_sync(mask, reinterpret_cast<unsigned long long>(scores), srcLaneIdx));
            this->ownsInstance = ::__shfl_sync(mask, ownsInstance, srcLaneIdx);
            this->pinned_instance = reinterpret_cast<decltype(pinned_instance)>(::__shfl_sync(mask, reinterpret_cast<unsigned long long>(pinned_instance), srcLaneIdx));
            this->d_instance = reinterpret_cast<decltype(d_instance)>(::__shfl_sync(mask, reinterpret_cast<unsigned long long>(d_instance), srcLaneIdx));
        }

        // Align sequence
        __host__ cost_t* align(const ::std::string& sequenceFileName);

        // Launch initialize kernel
        __host__ __forceinline__ void initialize() const {
            // Get block size
            const auto blockSize{KernelUtils::cudaSizeBlock<SequenceGraphKernels::initialize>()};

            // Get grid size
            const auto gridSize{KernelUtils::cudaSizeGrid(pangenomeGraph.getNumNodes(), blockSize)};

            // Launch kernel
            const auto initialSequenceBase{sequence[0]};
            KernelUtils::cudaLaunchKernel<SequenceGraphKernels::initialize>(gridSize, blockSize, 0, cudaStreamDefault, pinned_instance->pangenomeGraph.getBaseValues().getBases(), pinned_instance->costsDoubleBuffer.current(), pinned_instance->pangenomeGraph.getNumNodes(), initialSequenceBase);
        }

        // Launch substitutions kernel
        __host__ __forceinline__ void substitutions(const sequenceSize_t layerIdx, bool* __restrict__ const d_needsVisiting) const {
            // Get block size
            const auto blockSize{KernelUtils::cudaSizeBlock<SequenceGraphKernels::substitutions>()};

            // Get grid size
            const auto gridSize{KernelUtils::cudaSizeGrid(pangenomeGraph.getNumNodes(), blockSize)};

            // Launch kernel
            const auto sequenceBase{sequence[layerIdx]};
            KernelUtils::cudaLaunchKernel<SequenceGraphKernels::substitutions>(gridSize, blockSize, 0, cudaStreamDefault, pinned_instance->pangenomeGraph.getNeighborOffsets(), pinned_instance->pangenomeGraph.getNeighborValues(), pinned_instance->pangenomeGraph.getBaseValues().getBases(), pinned_instance->costsDoubleBuffer.alternate(), pinned_instance->costsDoubleBuffer.current(), d_needsVisiting, pinned_instance->connectedComponentsReverseMappings[static_cast<DNABase_t>(sequenceBase)], pinned_instance->pangenomeGraph.getNumNodes(), sequenceBase);
        }

        // Launch deletions kernel
        __host__ __forceinline__ void deletions() const {
            // Get block size
            const auto blockSize{KernelUtils::cudaSizeBlock<SequenceGraphKernels::deletions>()};

            // Get grid size
            const auto gridSize{KernelUtils::cudaSizeGrid(pangenomeGraph.getNumNodes(), blockSize)};

            // Launch kernel
            KernelUtils::cudaLaunchKernel<SequenceGraphKernels::deletions>(gridSize, blockSize, 0, cudaStreamDefault, pinned_instance->costsDoubleBuffer.alternate(), pinned_instance->costsDoubleBuffer.current(), pinned_instance->pangenomeGraph.getNumNodes());
        }

        // Launch insertions and propagations kernel
        __host__ __forceinline__ void insertionsAndPropagations(const sequenceSize_t layerIdx, nodeSize_t* __restrict__ const d_buffers, bool* __restrict__ const d_needsVisiting) const {
            // Cached block sizes
            static int cachedBlockSizes[NUM_BASES]{};

            // Define shared memory calculator
            const auto sequenceBase{sequence[layerIdx]};
            const auto maxConnectedComponentsSize{maxConnectedComponentsSizes[static_cast<DNABase_t>(sequenceBase)]};
            const auto SMemCalculator = [maxConnectedComponentsSize] __host__ __device__ (const targetSize_t blockSize) {
                // Get number of warps in the block
                const auto numWarps{(blockSize + KernelUtils::WARP_SIZE - 1) >> KernelUtils::WARP_SHIFT};

                return numWarps * (DoubleBuffer<nodeSize_t>::NUM_DOUBLE_BUFFERS * SequenceGraphKernels::SHARED_FRONTIER_BUFFER_SIZE * sizeof(nodeSize_t) + ((maxConnectedComponentsSize + Frontier::PACKING_FACTOR - 1) >> Frontier::PACK_SHIFT) * sizeof(queuePack_t));
            };

            // Check if block size has already been computed for the given DNA base
            auto blockSize{cachedBlockSizes[static_cast<DNABase_t>(sequenceBase)]};
            if (!blockSize) {
                // Get block size and round it down to be a multiple of WARP_SIZE
                int minGridSize{0};
                CUDA_CHECK(::cudaOccupancyMaxPotentialBlockSizeVariableSMem(&minGridSize, &blockSize, SequenceGraphKernels::insertionsAndPropagations, SMemCalculator, 0));
                blockSize &= ~(KernelUtils::WARP_SIZE - 1);
                blockSize = (blockSize < KernelUtils::WARP_SIZE) ? KernelUtils::WARP_SIZE : blockSize;

                // Cache block size
                cachedBlockSizes[static_cast<DNABase_t>(sequenceBase)] = blockSize;
            }

            // Get grid size (one warp per connected component)
            const auto numConnectedComponents{this->numConnectedComponents[static_cast<::size_t>(sequenceBase)]};
            const auto numWarpsPerBlock{(blockSize + KernelUtils::WARP_SIZE - 1) >> KernelUtils::WARP_SHIFT};
            const auto gridSize{KernelUtils::cudaSizeGrid(numConnectedComponents, numWarpsPerBlock)};

            // Get dynamic shared memory size
            const auto dynamicSMemSize{SMemCalculator(blockSize)};

            // Launch kernel
            const auto maxConnectedComponentSize{maxConnectedComponentsSizes[static_cast<DNABase_t>(sequenceBase)]};
            KernelUtils::cudaLaunchKernel<SequenceGraphKernels::insertionsAndPropagations>(gridSize, blockSize, dynamicSMemSize, cudaStreamDefault, pinned_instance->pangenomeGraph.getNeighborOffsets(), pinned_instance->pangenomeGraph.getNeighborValues(), pinned_instance->connectedComponentsOffsets[static_cast<DNABase_t>(sequenceBase)], pinned_instance->connectedComponentsMappings[static_cast<DNABase_t>(sequenceBase)], pinned_instance->connectedComponentsLocalIndexMappings[static_cast<DNABase_t>(sequenceBase)], pinned_instance->costsDoubleBuffer.current(), d_buffers, d_needsVisiting, numWarpsPerBlock, numConnectedComponents, maxConnectedComponentSize);
        }

        // Launch min cost kernel
        __host__ __forceinline__ void minCost(const sequenceSize_t scoreIdx) const {
            // Cached block sizes
            static int cachedBlockSize{0};

            // Define shared memory calculator
            const auto SMemCalculator = [] __host__ __device__ (const targetSize_t blockSize) {
                // Get number of warps in the block
                const auto numWarps{(blockSize + KernelUtils::WARP_SIZE - 1) >> KernelUtils::WARP_SHIFT};

                return numWarps * sizeof(cost_t);
            };

            // Check if block size has already been computed
            auto blockSize{cachedBlockSize};
            if (!blockSize) {
                // Get block size and round it down to be a multiple of WARP_SIZE
                int minGridSize{0};
                CUDA_CHECK(::cudaOccupancyMaxPotentialBlockSizeVariableSMem(&minGridSize, &blockSize, SequenceGraphKernels::minCost, SMemCalculator, 0));
                blockSize &= ~(KernelUtils::WARP_SIZE - 1);
                blockSize = (blockSize < KernelUtils::WARP_SIZE) ? KernelUtils::WARP_SIZE : blockSize;

                // Cache block size
                cachedBlockSize = blockSize;
            }

            // Get grid size
            const auto gridSize{KernelUtils::cudaSizeGrid(pangenomeGraph.getNumNodes(), cachedBlockSize)};

            // Get dynamic shared memory size
            const auto dynamicSMemSize{SMemCalculator(cachedBlockSize)};

            // Launch kernel
            KernelUtils::cudaLaunchKernel<SequenceGraphKernels::minCost>(gridSize, cachedBlockSize, dynamicSMemSize, cudaStreamDefault, pinned_instance->costsDoubleBuffer.current(), pinned_instance->scores, pinned_instance->pangenomeGraph.getNumNodes(), scoreIdx);
        }

    private:
        // Sequence graph implementation
        // NOTE: Uses pinned memory and linearized memory layout on the device memory
        PangenomeGraph pangenomeGraph{};
        PackedDNASequence sequence{};
        connectedComponentSize_t numConnectedComponents[NUM_BASES]{};
        nodeSize_t* __restrict__ connectedComponentsOffsets[NUM_BASES]{nullptr};
        nodeSize_t* __restrict__ connectedComponentsMappings[NUM_BASES]{nullptr};
        connectedComponentSize_t* __restrict__ connectedComponentsReverseMappings[NUM_BASES]{nullptr};
        connectedComponentSize_t* __restrict__ connectedComponentsLocalIndexMappings[NUM_BASES]{nullptr};
        connectedComponentSize_t maxConnectedComponentsSizes[NUM_BASES]{};
        DoubleBuffer<cost_t> costsDoubleBuffer{};
        scoreSize_t numScores{0};
        cost_t* __restrict__ scores{nullptr};
        bool ownsInstance{false};
        SequenceGraph* __restrict__ pinned_instance{nullptr};
        SequenceGraph* __restrict__ d_instance{nullptr};
    };

    // Insertions and propagations helper function
    __device__ __forceinline__ void SequenceGraphKernels::syncFrontierSizes(Frontier* __restrict__ const shared_frontier, Frontier* __restrict__ const warpFrontier, const unsigned mask) {
        shared_frontier->setQueueSize(::__reduce_max_sync(mask, shared_frontier->getQueueSize()));
        warpFrontier->setQueueSize(::__reduce_max_sync(mask, warpFrontier->getQueueSize()));
    }

    // Insertions and propagations helper function
    __device__ __forceinline__ void SequenceGraphKernels::processNeighbor(Frontier* __restrict__ const warpFrontier, Frontier* __restrict__ const shared_frontier, cost_t* __restrict__ const d_currentCosts, const nodeSize_t neighborIdx, const nodeSize_t neighborLocalIdx, const cost_t updatedCurrentLayerNeighborCost) {
        // Set cost to atomic min and get previous current layer neighbor cost
        // NOTE: Because in-degree for a node should be low, we can avoid doing warp / block level reduction in order to reduce overhead
        const auto previousCurrentLayerNeighborCost{::atomicMin(d_currentCosts + neighborIdx, updatedCurrentLayerNeighborCost)};

        // Check for improvement and that node index is node in the frontier already
        bool wantsToInsert{(updatedCurrentLayerNeighborCost < previousCurrentLayerNeighborCost) && !shared_frontier->isNodeInQueue(neighborLocalIdx)};

        // Get active mask
        const auto activeMask{::__activemask()};

        // Get matching mask
        const unsigned matchingMask{::__match_any_sync(activeMask, neighborIdx)};

        // Get wants to insert mask
        unsigned wantsToInsertMask{::__ballot_sync(activeMask, wantsToInsert)};

        // Deduplicate frontier queue insertions: only thread with lowest thread index inserts
        wantsToInsertMask &= matchingMask;
        if (wantsToInsert && (wantsToInsertMask & KernelUtils::lanemask_lt()) != 0) {
            wantsToInsert = false;
        }

        // Get updated wants to insert mask
        wantsToInsertMask = ::__ballot_sync(activeMask, wantsToInsert);

        // Get number of insertions
        const auto numInsertions{::__popc(wantsToInsertMask)};

        // Get old queue sizes
        const auto oldSharedQueueSize{shared_frontier->getQueueSize()};
        const auto oldWarpQueueSize{warpFrontier->getQueueSize()};

        // Calculate updates distributions
        const auto freeSharedSlots{SHARED_FRONTIER_BUFFER_SIZE - oldSharedQueueSize};
        const auto numInsertionsInShared{::min(numInsertions, freeSharedSlots)};

        // Grow queue sizes
        shared_frontier->growQueueSize(numInsertionsInShared);
        warpFrontier->growQueueSize(numInsertions - numInsertionsInShared);

        // Get insertion offset and check if inserting in shared or global frontier
        if (wantsToInsert) {
            if (const auto insertionOffset{static_cast<nodeSize_t>(::__popc(wantsToInsertMask & KernelUtils::lanemask_lt()))}; insertionOffset < numInsertionsInShared) {
                // Insert node into next shared frontier
                shared_frontier->atomicInsertNodeInQueue(neighborLocalIdx, oldSharedQueueSize + insertionOffset);
            } else {
                // Insert node into next global frontier
                warpFrontier->atomicInsertNodeInQueue(neighborLocalIdx, oldWarpQueueSize + (insertionOffset - numInsertionsInShared));
            }
        }
    }

    // Insertions and propagations helper function
    __device__ __forceinline__ void SequenceGraphKernels::processNode(const nodeSize_t nodeIdx, Frontier* __restrict__ const warpFrontier, Frontier* __restrict__ const shared_frontier, const edgeSize_t* __restrict__ const d_neighborOffsets, const nodeSize_t* __restrict__ const d_neighborValues, const connectedComponentSize_t* __restrict__ const d_connectedComponentLocalIndexMappings, cost_t* __restrict__ const d_currentCosts) {
        // Get updated current layer neighbor cost
        const auto updatedCurrentLayerNeighborCost{d_currentCosts[nodeIdx] + SequenceGraph::INSERTION_COST};

        // Loop over all neighbors
        const auto neighborsStart{d_neighborOffsets[nodeIdx]};
        const auto neighborEnd{d_neighborOffsets[nodeIdx + 1]};
        for (auto neighborOffset{neighborsStart}; neighborOffset < neighborEnd; ++neighborOffset) {
            // Get neighbor index
            const auto neighborIdx{d_neighborValues[neighborOffset]};
            const auto neighborLocalIdx{d_connectedComponentLocalIndexMappings[neighborIdx]};

            // Process neighbor
            processNeighbor(warpFrontier, shared_frontier, d_currentCosts, neighborIdx, neighborLocalIdx, updatedCurrentLayerNeighborCost);
        }

        // Sync frontier sizes
        syncFrontierSizes(shared_frontier, warpFrontier, ::__activemask());
    }
} // cuSGA

#endif //CUSGA_SEQUENCEGRAPH_CUH
