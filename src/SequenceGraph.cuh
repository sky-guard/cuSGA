#ifndef CUSGA_SEQUENCEGRAPH_CUH
#define CUSGA_SEQUENCEGRAPH_CUH

#include <cooperative_groups.h>

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
        inline constexpr targetSize_t SUB_SHARED_QUEUE_BUFFER_SIZE{((1 << 14) - 2) / sizeof(nodeSize_t)};
        __global__ void substitutions(const edgeSize_t* __restrict__ d_neighborOffsets, const nodeSize_t* __restrict__ d_neighborValues, const sequencePack_t* __restrict__ d_baseValues, const cost_t* __restrict__ d_previousCosts, cost_t* __restrict__ d_currentCosts, bool* __restrict__ d_needsVisiting, const connectedComponentSize_t* __restrict__ connectedComponentsReverseMapping, nodeSize_t numNodes, DNABase sequenceBase);
        __global__ void cooperativeSubstitutions(const edgeSize_t* __restrict__ d_neighborOffsets, const nodeSize_t* __restrict__ d_neighborValues, const sequencePack_t* __restrict__ d_baseValues, const cost_t* __restrict__ d_previousCosts, cost_t* __restrict__ d_currentCosts, nodeSize_t* __restrict__ d_frontierBufferSize, nodeSize_t* __restrict__ d_frontierBuffer, int* __restrict__ d_frontierQueue, nodeSize_t numNodes, DNABase sequenceBase);
        __global__ void cooperativeBlockAggregationSubstitutions(const edgeSize_t* __restrict__ d_neighborOffsets, const nodeSize_t* __restrict__ d_neighborValues, const sequencePack_t* __restrict__ d_baseValues, const cost_t* __restrict__ d_previousCosts, cost_t* __restrict__ d_currentCosts, nodeSize_t* __restrict__ d_frontierBufferSize, nodeSize_t* __restrict__ d_frontierBuffer, int* __restrict__ d_frontierQueue, nodeSize_t numNodes, DNABase sequenceBase);

        // Deletions kernel
        __global__ void deletions(const cost_t* __restrict__ d_previousCosts, cost_t* __restrict__ d_currentCosts, nodeSize_t numNodes);
        __global__ void cooperativeDeletions(const cost_t* __restrict__ d_previousCosts, cost_t* __restrict__ d_currentCosts, nodeSize_t* __restrict__ d_frontierQueueSizes, nodeSize_t numNodes);

        // Insertions and propagations kernel
        inline constexpr ::uint8_t INS_NUM_SIZES{4};
        inline constexpr targetSize_t INS_SHARED_FRONTIER_BUFFER_SIZE{KernelUtils::WARP_SIZE << 1};
        inline constexpr targetSize_t INS_SHARED_QUEUE_BUFFER_SIZE{((1 << 14) - 2) / sizeof(nodeSize_t)};
        __device__ __forceinline__ void processNeighbor(nodeSize_t* __restrict__ shared_queueBuffer, nodeSize_t* __restrict__ d_queueBuffer, nodeSize_t* __restrict__ shared_queueSize, nodeSize_t* __restrict__ shared_deviceQueueSize, queuePack_t* __restrict__ shared_isInQueue, cost_t* __restrict__ d_currentCosts, nodeSize_t neighborIdx, nodeSize_t neighborLocalIdx, cost_t updatedCurrentLayerNeighborCost);
        __device__ __forceinline__ void processNode(nodeSize_t* __restrict__ shared_queueBuffer, nodeSize_t* __restrict__ d_queueBuffer, nodeSize_t* __restrict__ shared_queueSize, nodeSize_t* __restrict__ shared_deviceQueueSize, queuePack_t* __restrict__ shared_isInQueue, cost_t* __restrict__ d_currentCosts, const edgeSize_t* __restrict__ d_neighborOffsets, const nodeSize_t* __restrict__ d_neighborValues, const connectedComponentSize_t* __restrict__ d_connectedComponentLocalIndexMappings, nodeSize_t nodeIdx);
        __global__ void insertionsAndPropagations(const edgeSize_t* __restrict__ d_neighborOffsets, const nodeSize_t* __restrict__ d_neighborValues, const nodeSize_t* __restrict__ d_connectedComponentOffsets, const nodeSize_t* __restrict__ d_connectedComponentMappings, const connectedComponentSize_t* __restrict__ d_connectedComponentLocalIndexMappings, cost_t* __restrict__ d_currentCosts, nodeSize_t* __restrict__ d_buffers, bool* __restrict__ d_needsVisiting, targetSize_t numWarpsPerBlock, connectedComponentSize_t numConnectedComponents, connectedComponentSize_t maxConnectedComponentSize, bool earlyExit);
        __global__ void cooperativeInsertionsAndPropagations(const edgeSize_t* __restrict__ d_neighborOffsets, const nodeSize_t* __restrict__ d_neighborValues, cost_t* __restrict__ d_currentCosts, nodeSize_t* __restrict__ d_frontierBuffer1, nodeSize_t* __restrict__ d_frontierBuffer2, int* __restrict__ d_frontierQueue1, int* __restrict__ d_frontierQueue2, nodeSize_t* __restrict__ d_frontierBufferSize11, nodeSize_t* __restrict__ d_frontierBufferSize12, nodeSize_t* __restrict__ d_frontierBufferSize21, nodeSize_t* __restrict__ d_frontierBufferSize22, bool earlyExit);
        __global__ void cooperativeBlockAggregationInsertionsAndPropagations(const edgeSize_t* __restrict__ d_neighborOffsets, const nodeSize_t* __restrict__ d_neighborValues, cost_t* __restrict__ d_currentCosts, nodeSize_t* __restrict__ d_frontierBuffer1, nodeSize_t* __restrict__ d_frontierBuffer2, int* __restrict__ d_frontierQueue1, int* __restrict__ d_frontierQueue2, nodeSize_t* __restrict__ d_frontierBufferSize11, nodeSize_t* __restrict__ d_frontierBufferSize12, nodeSize_t* __restrict__ d_frontierBufferSize21, nodeSize_t* __restrict__ d_frontierBufferSize22, bool earlyExit);

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
        __host__ __device__ __forceinline__ static void growBuffers(KernelUtils::BumpPtrAllocator* allocator, const nodeSize_t numNodes, const edgeSize_t numEdges, const sequenceSize_t maxSequenceLength, const connectedComponentSize_t totalNumConnectedComponents, const scoreSize_t numScores, const bool useConnectedComponents, const bool useCharacterGraphs) {
            // Grow size for pangenome graph
            PangenomeGraph::growBuffers(allocator, numNodes, numEdges);

            // Grow size for sequence
            PackedDNASequence::growBuffers(allocator, maxSequenceLength);

            if (useConnectedComponents) {
                // Grow size for connected components offsets
                allocator->grow<::std::remove_reference_t<decltype(connectedComponentsOffsets[0])>>(totalNumConnectedComponents + NUM_BASES);

                // Grow size for connected components mappings
                allocator->grow<::std::remove_reference_t<decltype(connectedComponentsMappings[0])>>(NUM_BASES * numNodes);

                // Grow size for connected components reverse mappings
                allocator->grow<::std::remove_reference_t<decltype(connectedComponentsReverseMappings[0])>>(NUM_BASES * numNodes);

                // Grow size for connected components local index mappings
                allocator->grow<::std::remove_reference_t<decltype(connectedComponentsLocalIndexMappings[0])>>(NUM_BASES * numNodes);

                if (useCharacterGraphs) {
                    // Grow size for connected components row offsets
                    allocator->grow<::std::remove_reference_t<decltype(connectedComponentsRowOffsets[0])>>(NUM_BASES * (numNodes + 1));

                    // Grow size for connected components column values
                    allocator->grow<::std::remove_reference_t<decltype(connectedComponentsColumnValues[0])>>(NUM_BASES * numEdges);
                }
            }

            // Grow size for costs double buffer
            DoubleBuffer<cost_t>::growBuffers(allocator, numNodes);

            // Grow size for scores
            allocator->grow<::std::remove_reference_t<decltype(scores[0])>>(numScores);
        }

        // Grow allocator using the expected buffers size, excluding the sequence
        // NOTE: There is no need to copy the sequence over to the device!
        __host__ __device__ __forceinline__ static void growBuffersWithoutSequence(KernelUtils::BumpPtrAllocator* allocator, const nodeSize_t numNodes, const edgeSize_t numEdges, const connectedComponentSize_t totalNumConnectedComponents, const scoreSize_t numScores, const bool useConnectedComponents, const bool useCharacterGraphs) {
            // Grow size for pangenome graph
            PangenomeGraph::growBuffers(allocator, numNodes, numEdges);

            if (useConnectedComponents) {
                // Grow size for connected components offsets
                allocator->grow<::std::remove_reference_t<decltype(connectedComponentsOffsets[0])>>(totalNumConnectedComponents + NUM_BASES);

                // Grow size for connected components mappings
                allocator->grow<::std::remove_reference_t<decltype(connectedComponentsMappings[0])>>(NUM_BASES * numNodes);

                // Grow size for connected components reverse mappings
                allocator->grow<::std::remove_reference_t<decltype(connectedComponentsReverseMappings[0])>>(NUM_BASES * numNodes);

                // Grow size for connected components local index mappings
                allocator->grow<::std::remove_reference_t<decltype(connectedComponentsLocalIndexMappings[0])>>(NUM_BASES * numNodes);

                if (useCharacterGraphs) {
                    // Grow size for connected components row offsets
                    allocator->grow<::std::remove_reference_t<decltype(connectedComponentsRowOffsets[0])>>(NUM_BASES * (numNodes + 1));

                    // Grow size for connected components column values
                    allocator->grow<::std::remove_reference_t<decltype(connectedComponentsColumnValues[0])>>(NUM_BASES * numEdges);
                }
            }

            // Grow size for costs double buffer
            DoubleBuffer<cost_t>::growBuffers(allocator, numNodes);

            // Grow size for scores
            allocator->grow<::std::remove_reference_t<decltype(scores[0])>>(numScores);
        }

        // Default constructor
        SequenceGraph() = default;
        // Parameterized constructor
        __host__ SequenceGraph(const ::std::string& pangenomeGraphFileName, const ::std::string& sequenceFileName, ::std::string const (& connectedComponentsFileNames)[NUM_BASES], bool useConnectedComponents, bool useCharacterGraphs, bool ownsInstance, SequenceGraph* pinned_instanceOptional = nullptr, KernelUtils::BumpPtrAllocator* allocatorOptional = nullptr);

        // Sequence graph constructor
        __host__ __device__ __forceinline__ SequenceGraph(const PangenomeGraph& pangenomeGraph, const PackedDNASequence& sequence, connectedComponentSize_t const (& numConnectedComponents)[NUM_BASES], nodeSize_t* const (& connectedComponentsOffsets)[NUM_BASES], nodeSize_t* const (& connectedComponentsMappings)[NUM_BASES], connectedComponentSize_t* const (& connectedComponentsReverseMappings)[NUM_BASES], connectedComponentSize_t* const (& connectedComponentsLocalIndexMappings)[NUM_BASES], connectedComponentSize_t const (& maxConnectedComponentsSizes)[NUM_BASES], edgeSize_t const (& connectedComponentsNumEdges)[NUM_BASES], edgeSize_t* const (& connectedComponentsRowOffsets)[NUM_BASES], nodeSize_t* const (& connectedComponentsColumnValues)[NUM_BASES], const DoubleBuffer<cost_t>& costsDoubleBuffer, const scoreSize_t numScores, cost_t* const scores, const bool ownsInstance, SequenceGraph* const pinned_instance = nullptr, SequenceGraph* const d_instance = nullptr) : pangenomeGraph{pangenomeGraph}, sequence{sequence}, costsDoubleBuffer{costsDoubleBuffer}, numScores{numScores}, scores{scores}, ownsInstance{ownsInstance}, pinned_instance{pinned_instance}, d_instance{d_instance} {
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

                // Set connected component num edges
                this->connectedComponentsNumEdges[baseIdx] = connectedComponentsNumEdges[baseIdx];

                // Set connected component row offsets
                this->connectedComponentsRowOffsets[baseIdx] = connectedComponentsRowOffsets[baseIdx];

                // Set connected component column values
                this->connectedComponentsColumnValues[baseIdx] = connectedComponentsColumnValues[baseIdx];
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
        __host__ SequenceGraph copyToDevice(bool useConnectedComponents, bool useCharacterGraphs, SequenceGraph* d_instanceOptional = nullptr, KernelUtils::BumpPtrAllocator* allocatorOptional = nullptr);
        // Free sequence graph
        __host__ void free() const;

        // Get pangenome graph
        __host__ __device__ __forceinline__ const PangenomeGraph& getPangenomeGraph() const {
            return pangenomeGraph;
        }

        // Get sequence
        __host__ __device__ __forceinline__ const PackedDNASequence& getSequence() const {
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
        __host__ __device__ __forceinline__ connectedComponentSize_t* getConnectedComponentReverseMappings(const DNABase characterGraphBase) const {
            return connectedComponentsReverseMappings[static_cast<DNABase_t>(characterGraphBase)];
        }

        // Get connected component reverse mapping for a given character graph DNA base and node index
        __host__ __device__ __forceinline__ connectedComponentSize_t getConnectedComponentReverseMapping(const DNABase characterGraphBase, const nodeSize_t nodeIdx) const {
            return connectedComponentsReverseMappings[static_cast<DNABase_t>(characterGraphBase)][nodeIdx];
        }

        // Get connected component local index mapping for a given character graph DNA base and node index
        __host__ __device__ __forceinline__ connectedComponentSize_t getConnectedComponentLocalIndexMapping(const DNABase characterGraphBase, const nodeSize_t nodeIdx) const {
            return connectedComponentsLocalIndexMappings[static_cast<DNABase_t>(characterGraphBase)][nodeIdx];
        }

        // Get max connected component size for a given character graph DNA base
        __host__ __device__ __forceinline__ connectedComponentSize_t getMaxConnectedComponentSize(const DNABase characterGraphBase) const {
            return maxConnectedComponentsSizes[static_cast<DNABase_t>(characterGraphBase)];
        }

        // Get connected component number of edges for a given character graph DNA base
        __host__ __device__ __forceinline__ edgeSize_t getConnectedComponentNumEdges(const DNABase characterGraphBase) const {
            return connectedComponentsNumEdges[static_cast<DNABase_t>(characterGraphBase)];
        }

        // Get connected component row offsets for a given character graph DNA base
        __host__ __device__ __forceinline__ edgeSize_t* getConnectedComponentRowOffsets(const DNABase characterGraphBase) const {
            return connectedComponentsRowOffsets[static_cast<DNABase_t>(characterGraphBase)];
        }

        // Get connected component column values for a given character graph DNA base
        __host__ __device__ __forceinline__ nodeSize_t* getConnectedComponentColumnValues(const DNABase characterGraphBase) const {
            return connectedComponentsColumnValues[static_cast<DNABase_t>(characterGraphBase)];
        }

        // Get costs double buffer
        __host__ __device__ __forceinline__ const DoubleBuffer<cost_t>& getCostsDoubleBuffer() const {
            return costsDoubleBuffer;
        }

        // Get number of scores
        __host__ __device__ __forceinline__ scoreSize_t getNumScores() const {
            return numScores;
        }

        // Get scores
        __host__ __device__ __forceinline__ cost_t* getScores() const {
            return scores;
        }

        // Copy back scores from device
        __host__ __forceinline__ cost_t* h2d_getScores() const {
            if (d_instance) {
                CUDA_CHECK(::cudaMemcpy(scores, pinned_instance->scores, numScores * sizeof(scores[0]),::cudaMemcpyDeviceToHost));
            }

            return scores;
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
                this->connectedComponentsNumEdges[baseIdx] = ::__shfl_sync(mask, connectedComponentsNumEdges[baseIdx], srcLaneIdx);
                this->connectedComponentsRowOffsets[baseIdx] = reinterpret_cast<::std::remove_reference_t<decltype(connectedComponentsRowOffsets[0])>>(::__shfl_sync(mask, reinterpret_cast<unsigned long long>(connectedComponentsRowOffsets[baseIdx]), srcLaneIdx));
                this->connectedComponentsColumnValues[baseIdx] = reinterpret_cast<::std::remove_reference_t<decltype(connectedComponentsColumnValues[0])>>(::__shfl_sync(mask, reinterpret_cast<unsigned long long>(connectedComponentsColumnValues[baseIdx]), srcLaneIdx));
            }
            this->costsDoubleBuffer.shuffle_sync(mask, srcLaneIdx);
            this->numScores = ::__shfl_sync(mask, numScores, srcLaneIdx);
            this->scores = reinterpret_cast<decltype(scores)>(::__shfl_sync(mask, reinterpret_cast<unsigned long long>(scores), srcLaneIdx));
            this->ownsInstance = ::__shfl_sync(mask, ownsInstance, srcLaneIdx);
            this->pinned_instance = reinterpret_cast<decltype(pinned_instance)>(::__shfl_sync(mask, reinterpret_cast<unsigned long long>(pinned_instance), srcLaneIdx));
            this->d_instance = reinterpret_cast<decltype(d_instance)>(::__shfl_sync(mask, reinterpret_cast<unsigned long long>(d_instance), srcLaneIdx));
        }

        // Align sequence using connected components
        __host__ cost_t* connectedComponentsAlign(const ::std::string& sequenceFileName, bool useCharacterGraphs);
        // Align sequence using grid
        __host__ cost_t* gridAlign(const ::std::string& sequenceFileName);
        // Align sequence using grid (with block aggregation)
        __host__ cost_t* gridBlockAggregationAlign(const ::std::string& sequenceFileName);

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
        __host__ __forceinline__ void substitutions(const sequenceSize_t layerIdx, bool* const d_needsVisiting) const {
            // Get block size
            const auto blockSize{KernelUtils::cudaSizeBlock<SequenceGraphKernels::substitutions>()};

            // Get grid size
            const auto gridSize{KernelUtils::cudaSizeGrid(pangenomeGraph.getNumNodes(), blockSize)};

            // Launch kernel
            const auto sequenceBase{sequence[layerIdx]};
            KernelUtils::cudaLaunchKernel<SequenceGraphKernels::substitutions>(gridSize, blockSize, 0, cudaStreamDefault, pinned_instance->pangenomeGraph.getNeighborOffsets(), pinned_instance->pangenomeGraph.getNeighborValues(), pinned_instance->pangenomeGraph.getBaseValues().getBases(), pinned_instance->costsDoubleBuffer.alternate(), pinned_instance->costsDoubleBuffer.current(), d_needsVisiting, pinned_instance->connectedComponentsReverseMappings[static_cast<DNABase_t>(sequenceBase)], pinned_instance->pangenomeGraph.getNumNodes(), sequenceBase);
        }

        // Launch cooperative substitutions kernel
        __host__ __forceinline__ void cooperativeSubstitutions(const sequenceSize_t layerIdx, nodeSize_t* const d_frontierBufferSize, nodeSize_t* const d_frontierBuffer, int* const d_frontierQueue) const {
            // Get block size
            const auto blockSize{KernelUtils::cudaSizeBlock<SequenceGraphKernels::cooperativeSubstitutions>()};

            // Get grid size
            const auto gridSize{KernelUtils::cudaSizeGrid(pangenomeGraph.getNumNodes(), blockSize)};

            // Launch kernel
            const auto sequenceBase{sequence[layerIdx]};
            KernelUtils::cudaLaunchKernel<SequenceGraphKernels::cooperativeSubstitutions>(gridSize, blockSize, 0, cudaStreamDefault, pinned_instance->pangenomeGraph.getNeighborOffsets(), pinned_instance->pangenomeGraph.getNeighborValues(), pinned_instance->pangenomeGraph.getBaseValues().getBases(), pinned_instance->costsDoubleBuffer.alternate(), pinned_instance->costsDoubleBuffer.current(), d_frontierBufferSize, d_frontierBuffer, d_frontierQueue, pinned_instance->pangenomeGraph.getNumNodes(), sequenceBase);
        }

        // Launch cooperative substitutions kernel (with block aggregation)
        __host__ __forceinline__ void cooperativeBlockAggregationSubstitutions(const sequenceSize_t layerIdx, nodeSize_t* const d_frontierBufferSize, nodeSize_t* const d_frontierBuffer, int* const d_frontierQueue) const {
            // Get block size
            const auto blockSize{KernelUtils::cudaSizeBlock<SequenceGraphKernels::cooperativeBlockAggregationSubstitutions>()};

            // Get grid size
            const auto gridSize{KernelUtils::cudaSizeGrid(pangenomeGraph.getNumNodes(), blockSize)};

            // Launch kernel
            const auto sequenceBase{sequence[layerIdx]};
            KernelUtils::cudaLaunchKernel<SequenceGraphKernels::cooperativeBlockAggregationSubstitutions>(gridSize, blockSize, 0, cudaStreamDefault, pinned_instance->pangenomeGraph.getNeighborOffsets(), pinned_instance->pangenomeGraph.getNeighborValues(), pinned_instance->pangenomeGraph.getBaseValues().getBases(), pinned_instance->costsDoubleBuffer.alternate(), pinned_instance->costsDoubleBuffer.current(), d_frontierBufferSize, d_frontierBuffer, d_frontierQueue, pinned_instance->pangenomeGraph.getNumNodes(), sequenceBase);
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

        // Launch cooperative deletions kernel
        __host__ __forceinline__ void cooperativeDeletions(nodeSize_t* const d_frontierQueueSizes) const {
            // Get block size
            const auto blockSize{KernelUtils::cudaSizeBlock<SequenceGraphKernels::cooperativeDeletions>()};

            // Get grid size
            const auto gridSize{KernelUtils::cudaSizeGrid(pangenomeGraph.getNumNodes(), blockSize)};

            // Launch kernel
            KernelUtils::cudaLaunchKernel<SequenceGraphKernels::cooperativeDeletions>(gridSize, blockSize, 0, cudaStreamDefault, pinned_instance->costsDoubleBuffer.alternate(), pinned_instance->costsDoubleBuffer.current(), d_frontierQueueSizes, pinned_instance->pangenomeGraph.getNumNodes());
        }

        // Launch insertions and propagations kernel
        __host__ __forceinline__ void insertionsAndPropagations(const sequenceSize_t layerIdx, nodeSize_t* const d_buffers, bool* const d_needsVisiting, const bool useCharacterGraphs) const {
            // Cached block sizes
            static int cachedBlockSizes[NUM_BASES]{};

            // Define shared memory calculator
            const auto sequenceBase{sequence[layerIdx]};
            const auto maxConnectedComponentsSize{maxConnectedComponentsSizes[static_cast<DNABase_t>(sequenceBase)]};
            const auto SMemCalculator = [maxConnectedComponentsSize] __host__ __device__ (const targetSize_t blockSize) {
                // Get number of warps in the block
                const auto numWarps{(blockSize + KernelUtils::WARP_SIZE - 1) >> KernelUtils::WARP_SHIFT};

                return numWarps * (SequenceGraphKernels::INS_NUM_SIZES * sizeof(nodeSize_t) + DoubleBuffer<nodeSize_t>::NUM_DOUBLE_BUFFERS * SequenceGraphKernels::INS_SHARED_FRONTIER_BUFFER_SIZE * sizeof(nodeSize_t) + ((maxConnectedComponentsSize + Frontier::PACKING_FACTOR - 1) >> Frontier::PACK_SHIFT) * sizeof(queuePack_t));
            };

            // Check if block size has already been computed for the given DNA base
            auto blockSize{cachedBlockSizes[static_cast<DNABase_t>(sequenceBase)]};
            if (!blockSize) {
                // Get block size and round it down to be a multiple of WARP_SIZE
                int minGridSize{0};
                CUDA_CHECK(::cudaOccupancyMaxPotentialBlockSizeVariableSMem(&minGridSize, &blockSize, SequenceGraphKernels::insertionsAndPropagations, SMemCalculator, 0));
                blockSize &= ~(KernelUtils::WARP_SIZE - 1);
                blockSize = (blockSize < KernelUtils::WARP_SIZE)? KernelUtils::WARP_SIZE : blockSize;

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
            KernelUtils::cudaLaunchKernel<SequenceGraphKernels::insertionsAndPropagations>(gridSize, blockSize, dynamicSMemSize, cudaStreamDefault, (useCharacterGraphs)? pinned_instance->connectedComponentsRowOffsets[static_cast<DNABase_t>(sequenceBase)] : pinned_instance->pangenomeGraph.getNeighborOffsets(), (useCharacterGraphs)? pinned_instance->connectedComponentsColumnValues[static_cast<DNABase_t>(sequenceBase)] : pinned_instance->pangenomeGraph.getNeighborValues(), pinned_instance->connectedComponentsOffsets[static_cast<DNABase_t>(sequenceBase)], pinned_instance->connectedComponentsMappings[static_cast<DNABase_t>(sequenceBase)], pinned_instance->connectedComponentsLocalIndexMappings[static_cast<DNABase_t>(sequenceBase)], pinned_instance->costsDoubleBuffer.current(), d_buffers, d_needsVisiting, numWarpsPerBlock, numConnectedComponents, maxConnectedComponentSize, layerIdx == (sequence.getNumBases() - 1));
        }

        // Launch cooperative insertions and propagations kernel
        __host__ __forceinline__ void cooperativeInsertionsAndPropagations(const sequenceSize_t layerIdx, nodeSize_t* const d_frontierBuffer1, nodeSize_t* const d_frontierBuffer2, int* const d_frontierQueue1, int* const d_frontierQueue2, nodeSize_t* const d_frontierBufferSize11, nodeSize_t* const d_frontierBufferSize12, nodeSize_t* const d_frontierBufferSize21, nodeSize_t* const d_frontierBufferSize22) const {
            // Cached grid and block sizes
            static int cachedGridSizes[NUM_BASES]{};
            static int cachedBlockSizes[NUM_BASES]{};

            // Get sequence base
            const auto sequenceBase{sequence[layerIdx]};

            // Check if sizes have already been computed for the given DNA base
            auto blockSize{cachedBlockSizes[static_cast<DNABase_t>(sequenceBase)]};
            if (!blockSize) {
                // Get block size and round it down to be a multiple of WARP_SIZE
                CUDA_CHECK(::cudaOccupancyMaxPotentialBlockSize(&cachedGridSizes[static_cast<DNABase_t>(sequenceBase)], &blockSize, SequenceGraphKernels::cooperativeInsertionsAndPropagations, 0, 0));
                blockSize &= ~(KernelUtils::WARP_SIZE - 1);
                blockSize = (blockSize < KernelUtils::WARP_SIZE)? KernelUtils::WARP_SIZE : blockSize;

                // Cache block size
                cachedBlockSizes[static_cast<DNABase_t>(sequenceBase)] = blockSize;
            }

            // Get grid size
            const auto gridSize{cachedGridSizes[static_cast<DNABase_t>(sequenceBase)]};

            // Launch kernel
            KernelUtils::cudaLaunchCooperativeKernel<SequenceGraphKernels::cooperativeInsertionsAndPropagations>(gridSize, blockSize, 0, cudaStreamDefault, pinned_instance->pangenomeGraph.getNeighborOffsets(), pinned_instance->pangenomeGraph.getNeighborValues(), pinned_instance->costsDoubleBuffer.current(), d_frontierBuffer1, d_frontierBuffer2, d_frontierQueue1, d_frontierQueue2, d_frontierBufferSize11, d_frontierBufferSize12, d_frontierBufferSize21, d_frontierBufferSize22, layerIdx == (sequence.getNumBases() - 1));
        }

        // Launch cooperative insertions and propagations kernel (with block aggregation)
        __host__ __forceinline__ void cooperativeBlockAggregationInsertionsAndPropagations(const sequenceSize_t layerIdx, nodeSize_t* const d_frontierBuffer1, nodeSize_t* const d_frontierBuffer2, int* const d_frontierQueue1, int* const d_frontierQueue2, nodeSize_t* const d_frontierBufferSize11, nodeSize_t* const d_frontierBufferSize12, nodeSize_t* const d_frontierBufferSize21, nodeSize_t* const d_frontierBufferSize22) const {
            // Cached grid and block sizes
            static int cachedGridSizes[NUM_BASES]{};
            static int cachedBlockSizes[NUM_BASES]{};

            // Get sequence base
            const auto sequenceBase{sequence[layerIdx]};

            // Check if sizes have already been computed for the given DNA base
            auto blockSize{cachedBlockSizes[static_cast<DNABase_t>(sequenceBase)]};
            if (!blockSize) {
                // Get block size and round it down to be a multiple of WARP_SIZE
                CUDA_CHECK(::cudaOccupancyMaxPotentialBlockSize(&cachedGridSizes[static_cast<DNABase_t>(sequenceBase)], &blockSize, SequenceGraphKernels::cooperativeBlockAggregationInsertionsAndPropagations, 0, 0));
                blockSize &= ~(KernelUtils::WARP_SIZE - 1);
                blockSize = (blockSize < KernelUtils::WARP_SIZE)? KernelUtils::WARP_SIZE : blockSize;

                // Cache block size
                cachedBlockSizes[static_cast<DNABase_t>(sequenceBase)] = blockSize;
            }

            // Get grid size
            const auto gridSize{cachedGridSizes[static_cast<DNABase_t>(sequenceBase)]};

            // Launch kernel
            KernelUtils::cudaLaunchCooperativeKernel<SequenceGraphKernels::cooperativeBlockAggregationInsertionsAndPropagations>(gridSize, blockSize, 0, cudaStreamDefault, pinned_instance->pangenomeGraph.getNeighborOffsets(), pinned_instance->pangenomeGraph.getNeighborValues(), pinned_instance->costsDoubleBuffer.current(), d_frontierBuffer1, d_frontierBuffer2, d_frontierQueue1, d_frontierQueue2, d_frontierBufferSize11, d_frontierBufferSize12, d_frontierBufferSize21, d_frontierBufferSize22, layerIdx == (sequence.getNumBases() - 1));
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
        nodeSize_t* connectedComponentsOffsets[NUM_BASES]{nullptr};
        nodeSize_t* connectedComponentsMappings[NUM_BASES]{nullptr};
        connectedComponentSize_t* connectedComponentsReverseMappings[NUM_BASES]{nullptr};
        connectedComponentSize_t* connectedComponentsLocalIndexMappings[NUM_BASES]{nullptr};
        connectedComponentSize_t maxConnectedComponentsSizes[NUM_BASES]{};
        edgeSize_t connectedComponentsNumEdges[NUM_BASES]{};
        edgeSize_t* connectedComponentsRowOffsets[NUM_BASES]{nullptr};
        nodeSize_t* connectedComponentsColumnValues[NUM_BASES]{nullptr};
        DoubleBuffer<cost_t> costsDoubleBuffer{};
        scoreSize_t numScores{0};
        cost_t* scores{nullptr};
        bool ownsInstance{false};
        SequenceGraph* pinned_instance{nullptr};
        SequenceGraph* d_instance{nullptr};
    };

    // Insertions and propagations helper function
    __device__ __forceinline__ void SequenceGraphKernels::processNeighbor(nodeSize_t* __restrict__ const shared_queueBuffer, nodeSize_t* __restrict__ const d_queueBuffer, nodeSize_t* __restrict__ const shared_queueSize, nodeSize_t* __restrict__ const shared_deviceQueueSize, queuePack_t* __restrict__ const shared_isInQueue, cost_t* __restrict__ const d_currentCosts, const nodeSize_t neighborIdx, const nodeSize_t neighborLocalIdx, const cost_t updatedCurrentLayerNeighborCost) {
        // Set cost to atomic min and get previous current layer neighbor cost and check for improvement
        // NOTE: Because in-degree for a node should be low, we can avoid doing warp / block level reduction in order to reduce overhead
        if (const auto previousCurrentLayerNeighborCost{::atomicMin(d_currentCosts + neighborIdx, updatedCurrentLayerNeighborCost)}; updatedCurrentLayerNeighborCost >= previousCurrentLayerNeighborCost) {
            return;
        }

        // Get pack index and bitmask
        const auto chunkIdx = neighborLocalIdx >> Frontier::PACK_SHIFT;
        const auto bitmask = Frontier::BITMASK << (neighborLocalIdx & (Frontier::PACKING_FACTOR - 1));

        // Perform atomic OR to set bit and check if node already present in the frontier
        if (const auto oldMask{::atomicOr(shared_isInQueue + chunkIdx, bitmask)}; (oldMask & bitmask) != 0) {
            return;
        }

        // Grow queue size and check for shared queue overflow
        if (const auto oldSharedQueueSize{::atomicAdd(shared_queueSize, 1)}; oldSharedQueueSize < INS_SHARED_FRONTIER_BUFFER_SIZE) {
            // Insert in shared queue
            shared_queueBuffer[oldSharedQueueSize] = neighborIdx;
        }
        else {
            // Fallback to global queue
            const auto oldGlobalQueueSize{::atomicAdd(shared_deviceQueueSize, 1)};
            d_queueBuffer[oldGlobalQueueSize] = neighborIdx;
        }
    }

    // Insertions and propagations helper function
    __device__ __forceinline__ void SequenceGraphKernels::processNode(nodeSize_t* __restrict__ const shared_queueBuffer, nodeSize_t* __restrict__ const d_queueBuffer, nodeSize_t* __restrict__ const shared_queueSize, nodeSize_t* __restrict__ const shared_deviceQueueSize, queuePack_t* __restrict__ const shared_isInQueue, cost_t* __restrict__ const d_currentCosts, const edgeSize_t* __restrict__ const d_neighborOffsets, const nodeSize_t* __restrict__ const d_neighborValues, const connectedComponentSize_t* __restrict__ const d_connectedComponentLocalIndexMappings, const nodeSize_t nodeIdx) {
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
            processNeighbor(shared_queueBuffer, d_queueBuffer, shared_queueSize, shared_deviceQueueSize, shared_isInQueue, d_currentCosts, neighborIdx, neighborLocalIdx, updatedCurrentLayerNeighborCost);
        }
    }
} // cuSGA

#endif //CUSGA_SEQUENCEGRAPH_CUH
