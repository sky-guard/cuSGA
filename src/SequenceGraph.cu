#include "SequenceGraph.cuh"

#include <fstream>

#include "KernelUtils.cuh"

namespace cuSGA {
    __host__ SequenceGraph::SequenceGraph(const ::std::string& pangenomeGraphFileName, ::std::string const (& connectedComponentsFileNames)[NUM_BASES], bool ownsInstance, SequenceGraph* pinned_instanceOptional, SequenceGraph* d_instanceOptional) {
        // Read sequence from file
        const auto sequence{PackedDNASequence::create()};

        // Read pangenome graph from file
        const auto pangenomeGraph{PangenomeGraph::createFromFile(pangenomeGraphFileName)};

        // Create costs double buffer instance
        const auto numNodes{pangenomeGraph->getNumNodes()};
        const auto costsDoubleBuffer{DoubleBuffer<::cuda::atomic<::uint64_t, ::cuda::thread_scope_device>>::create(numNodes)};

        // Read connected components from files
        ::size_t numConnectedComponents[NUM_BASES]{};
        ::size_t* connectedComponentsOffsets[NUM_BASES]{nullptr};
        ::size_t* connectedComponentsMappings[NUM_BASES]{nullptr};
#pragma unroll
        for (::size_t i{0}; i < NUM_BASES; ++i) {
            // Open connected components file
            const auto connectedComponentsFileName{connectedComponentsFileNames[i]};
            ::std::ifstream connectedComponentsFile{connectedComponentsFileName};
            if (!connectedComponentsFile.is_open()) {
                throw ::std::runtime_error{::std::format("Unable to open file: {}", connectedComponentsFileName)};
            }

            // Read number of connected components from file
            ::size_t numConnectedComponentsValue{0};
            if (!(connectedComponentsFile >> numConnectedComponentsValue)) {
                throw ::std::runtime_error{::std::format("An error occurred while reading values from file: {}", connectedComponentsFileName)};
            }
            numConnectedComponents[i] = numConnectedComponentsValue;

            // Read connected components offsets from file
            const auto connectedComponentsOffsetsValue{new ::size_t[numConnectedComponentsValue + 1]{}};
            for (::size_t j{0}; j < numConnectedComponentsValue + 1; ++j) {
                ::size_t connectedComponentsOffset{0};
                if (!(connectedComponentsFile >> connectedComponentsOffset)) {
                    throw ::std::runtime_error{::std::format("An error occurred while reading values from file: {}", connectedComponentsFileName)};
                }
                connectedComponentsOffsetsValue[j] = connectedComponentsOffset;
            }
            connectedComponentsOffsets[i] = connectedComponentsOffsetsValue;

            // Read connected components mappings from file
            const auto connectedComponentsMappingsValue{new ::size_t[numNodes]{}};
            for (::size_t j{0}; j < numNodes; ++j) {
                ::size_t connectedComponentsMapping{0};
                if (!(connectedComponentsFile >> connectedComponentsMapping)) {
                    throw ::std::runtime_error{::std::format("An error occurred while reading values from file: {}", connectedComponentsFileName)};
                }
                connectedComponentsMappingsValue[j] = connectedComponentsMapping;
            }
            connectedComponentsMappings[i] = connectedComponentsMappingsValue;
        }

        // Create sequence graph instance
        const auto sequenceGraph{new SequenceGraph{sequence, pangenomeGraph, numConnectedComponents, connectedComponentsOffsets, connectedComponentsMappings, costsDoubleBuffer}};

        return sequenceGraph;
    }

    __host__ SequenceGraph* SequenceGraph::copyToDevice(SequenceGraph* d_instanceOptional, KernelUtils::BumpPtrAllocator* allocatorOptional) {
        // Check if device instance already exists for this sequence graph
        if (d_instance) {
            return d_instance;
        }

        // Allocate device buffers
        const auto numNodes{pangenomeGraph->getNumNodes()};
        ::size_t* d_connectedComponentsOffsets[NUM_BASES]{nullptr};
        ::size_t* d_connectedComponentsMappings[NUM_BASES]{nullptr};
#pragma unroll
        for (::size_t i{0}; i < NUM_BASES; ++i) {
            KernelUtils::cudaMalloc(&d_connectedComponentsOffsets[i], (numConnectedComponents[i] + 1) * sizeof(::uint64_t));
            KernelUtils::cudaMalloc(&d_connectedComponentsMappings[i], numNodes * sizeof(::uint64_t));
        }

        // Copy buffers data from host to device
        const auto d_sequence{sequence->copyToDevice()};
        const auto d_pangenomeGraph{pangenomeGraph->copyToDevice()};
        const auto d_costsDoubleBuffer{costsDoubleBuffer->copyToDevice()};
#pragma unroll
        for (::size_t i{0}; i < NUM_BASES; ++i) {
            KernelUtils::cudaMemcpy(d_connectedComponentsOffsets[i], connectedComponentsOffsets[i], (numConnectedComponents[i] + 1) * sizeof(::uint64_t), ::cudaMemcpyHostToDevice);
            KernelUtils::cudaMemcpy(d_connectedComponentsMappings[i], connectedComponentsMappings[i], numNodes * sizeof(::uint64_t), ::cudaMemcpyHostToDevice);
        }

        // Allocate device pangenome graph instance
        SequenceGraph* d_sequenceGraph{nullptr};
        KernelUtils::cudaMalloc(&d_sequenceGraph, sizeof(SequenceGraph));

        // Create temporary host instance holding the device pointers
        const SequenceGraph deviceSequenceGraph{d_sequence, d_pangenomeGraph, numConnectedComponents, d_connectedComponentsOffsets, d_connectedComponentsMappings, d_costsDoubleBuffer, score.load(::cuda::memory_order_relaxed), d_instance};

        // Update host instance data
        this->d_instance = d_sequenceGraph;

        // Copy instance data from host to device
        KernelUtils::cudaMemcpy(d_sequenceGraph, &deviceSequenceGraph, sizeof(SequenceGraph), ::cudaMemcpyHostToDevice);

        return d_sequenceGraph;
    }

    __host__ void SequenceGraph::free() const {
        // Free device memory if device instance is present
        if (d_instance) {
            // Create a temporary host copy of the device instance to get its internal device pointers
            SequenceGraph deviceSequenceGraph{};
            KernelUtils::cudaMemcpy(&deviceSequenceGraph, d_instance, sizeof(SequenceGraph), ::cudaMemcpyDeviceToHost);

            // Free device connected components offsets and mappings
#pragma unroll
            for (::size_t i{0}; i < NUM_BASES; ++i) {
                KernelUtils::cudaFree(deviceSequenceGraph.connectedComponentsOffsets[i]);
                KernelUtils::cudaFree(deviceSequenceGraph.connectedComponentsMappings[i]);
            }

            // Free device sequence graph instance
            KernelUtils::cudaFree(d_instance);
        }

        // Free host memory
        if (sequence) {
            this->sequence->free();
        }
        if (pangenomeGraph) {
            this->pangenomeGraph->free(true);
        }
#pragma unroll
        for (::size_t i{0}; i < NUM_BASES; ++i) {
            delete[] connectedComponentsOffsets[i];
            delete[] connectedComponentsMappings[i];
        }
        if (costsDoubleBuffer) {
            this->costsDoubleBuffer->free();
        }
        delete this;
    }

    __host__ ::std::vector<uint64_t> SequenceGraph::align(const ::std::string& sequenceFileName) {
        // Open file
        ::std::ifstream sequenceFile{sequenceFileName};
        if (!sequenceFile.is_open()) {
            throw ::std::runtime_error{::std::format("Unable to open file: {}", sequenceFileName)};
        }

        // Create scores instance
        ::std::vector<::uint64_t> scores{};

        // Copy sequence graph instance to device
        copyToDevice();

        while (sequence->readFromFile(sequenceFileName, sequenceFile)) {
            // Check for non-empty sequence
            if (sequence->getNumBases() == 0) {
                throw ::std::runtime_error{"Unable to align an empty sequence!"};
            }

            // Perform initialization step
            initialize(false);

            // Swap costs double buffer for the next layer
            costsDoubleBuffer->h2d_swap();

            // Solve alignment layer by layer
            for (::size_t layerIdx{1}; layerIdx < sequence->getNumBases(); ++layerIdx) {
                // Perform deletions for the given layer
                deletions(layerIdx,false);

                // Perform substitutions for the given layer
                substitutions(layerIdx, false);

                // Skip insertions and propagations for the last layer
                if (layerIdx < sequence->getNumBases() - 1) {
                    // Perform insertions for the given layer
                    insertions(frontier, false);

                    // Perform propagations for the given layer
                    propagations(layerIdx, frontier, false);

                    // Swap costs double buffer for the next layer
                    costsDoubleBuffer->h2d_swap();
                }
            }

            // Compute minimum score
            minCost(true);

            // Copy back score from device
            const auto score{getScoreSync()};

            // Save score
            scores.emplace_back(score);

            // Reset score
            resetScoreSync();
        }

        // Close file
        sequenceFile.close();

        return scores;
    }

    __global__ void SequenceGraphKernels::initialize(const SequenceGraph d_sequenceGraph, const DNABase initialSequenceBase) { // NOLINT
        // Get thread node index and check for thread overflow
        if (const auto nodeIdx{::blockIdx.x * ::blockDim.x + ::threadIdx.x}; nodeIdx < d_sequenceGraph.getPangenomeGraph().getNumNodes()) {
            // Get DNA base for the current node
            const auto nodeBase{d_sequenceGraph.getPangenomeGraph().getDNABase(nodeIdx)};

            // Compute updated node cost
            const auto initialNodeCost{(initialSequenceBase == nodeBase)? 0 : SequenceGraph::INITIALIZATION_COST};

            // Initialize node cost for the next layer
            d_sequenceGraph.getCostsDoubleBuffer().current()[nodeIdx] = initialNodeCost;
        }
    }

    __global__ void SequenceGraphKernels::substitutions(const SequenceGraph d_sequenceGraph, const DNABase sequenceBase) { // NOLINT
        // Get thread node index and check for thread overflow
        if (const auto nodeIdx{::blockIdx.x * ::blockDim.x + ::threadIdx.x}; nodeIdx < d_sequenceGraph.getPangenomeGraph().getNumNodes()) {
            // Get node cost in the previous layer
            const auto previousLayerNodeCost{d_sequenceGraph.getCostsDoubleBuffer().alternate()[nodeIdx]};

            // Loop over all neighbors
            const auto neighborsStart{d_sequenceGraph.getPangenomeGraph().getNeighborsOffset(nodeIdx)};
            const auto neighborEnd{d_sequenceGraph.getPangenomeGraph().getNeighborsOffset(nodeIdx + 1)};
            for (auto neighborOffset{neighborsStart}; neighborOffset < neighborEnd; ++neighborOffset) {
                // Get neighbor index
                const auto neighborIdx{d_sequenceGraph.getPangenomeGraph().getNeighbor(neighborOffset)};

                // Get DNA base for the neighbor
                const auto neighborBase{d_sequenceGraph.getPangenomeGraph().getDNABase(neighborIdx)};

                // Compute updated node cost
                const auto updatedCurrentLayerNeighborCost{(sequenceBase == neighborBase)? previousLayerNodeCost : previousLayerNodeCost + SequenceGraph::SUBSTITUTION_COST};

                // Set cost to atomic min
                // NOTE: Because in-degree for a node should be low, we can avoid doing warp / block level reduction in order to reduce overhead
                ::atomicMin(&d_sequenceGraph.getCostsDoubleBuffer().current()[neighborIdx], updatedCurrentLayerNeighborCost);
            }
        }
    }

    __global__ void SequenceGraphKernels::deletions(const SequenceGraph d_sequenceGraph) { // NOLINT
        // Get thread node index and check for thread overflow
        if (const auto nodeIdx{::blockIdx.x * ::blockDim.x + ::threadIdx.x}; nodeIdx < d_sequenceGraph.getPangenomeGraph().getNumNodes()) {
            // Get node cost in the previous layer
            const auto previousLayerNodeCost{d_sequenceGraph.getCostsDoubleBuffer().alternate()[nodeIdx]};

            // Compute updated node cost
            const auto currentLayerNodeCost{previousLayerNodeCost + SequenceGraph::DELETION_COST};

            // Initialize node cost for the next layer
            d_sequenceGraph.getCostsDoubleBuffer().current()[nodeIdx] = currentLayerNodeCost;
        }
    }

    __device__ __forceinline__ void SequenceGraphKernels::processNeighbor(const SequenceGraph& d_sequenceGraph, Frontier* const warpFrontier, const nodeSize_t neighborIdx, const cost_t updatedCurrentLayerNeighborCost, const ::uint8_t laneIdx) {
        // Set cost to atomic min and get previous current layer neighbor cost
        // NOTE: Because in-degree for a node should be low, we can avoid doing warp / block level reduction in order to reduce overhead
        const auto previousCurrentLayerNeighborCost{::atomicMin(&d_sequenceGraph.getCostsDoubleBuffer().current()[neighborIdx], updatedCurrentLayerNeighborCost)};

        // Get active mask
        const auto activeMask{::__activemask()};

        // Check for improvement
        bool wantsToInsert{(updatedCurrentLayerNeighborCost < previousCurrentLayerNeighborCost) && warpFrontier->isNodeInQueue(neighborIdx)};

        // Deduplicate frontier queue insertions: only thread with lowest thread index inserts
        if (wantsToInsert) {
            // Finds a bitmask of all active threads in the warp that have the exact same neighbor index
            unsigned matchingNeighborIdxMask{0};
            asm volatile("match.any.sync.b32 %0, %1, %2;"
                        : "=r"(matchingNeighborIdxMask)
                        : "r"(neighborIdx), "r"(activeMask));

            // Get the lowest lane ID among the threads that matched with this neighbor index and set improved to false for others
            const auto lowestMatchingLane{::__ffs(static_cast<int>(matchingNeighborIdxMask)) - 1};
            wantsToInsert = (laneIdx == lowestMatchingLane);
        }

        // Get improved mask
        const auto improvedMask{::__ballot_sync(activeMask, wantsToInsert)};

        // Check if thread passed deduplication check
        if (wantsToInsert) {
            // Get insertion offset
            const auto insertionOffset{::__popc(improvedMask & ((1 << laneIdx) - 1))};

            // Insert node into next frontier
            warpFrontier->insertInQueue(neighborIdx, warpFrontier->getQueueSize() + insertionOffset);

            // Grow queue size
            warpFrontier->growQueueSize(::__popc(improvedMask));
        }
    }

    __global__ void SequenceGraphKernels::insertionsAndPropagations(const SequenceGraph d_sequenceGraph, const DNABase sequenceBase, const connectedComponentSize_t maxConnectedComponentSize, nodeSize_t* const d_buffers) { // NOLINT
        // Shared memory array to store the warp-level isInQueue
        extern __shared__ bool shared_isInQueue[];

        // Get thread warp ID and check for thread overflow
        if (const auto warpID{(::blockIdx.x * ::blockDim.x + ::threadIdx.x) >> KernelUtils::WARP_SHIFT}; warpID < d_sequenceGraph.getNumConnectedComponents(sequenceBase)) {
            // Get warp index and lane index
            const auto warpIdx{::threadIdx.x >> KernelUtils::WARP_SHIFT};
            const auto laneIdx{::threadIdx.x & (KernelUtils::WARP_SIZE - 1)};

            // Get connected component details
            const auto connectedComponentStart{d_sequenceGraph.getConnectedComponentOffset(sequenceBase, warpID)};
            const auto connectedComponentEnd{d_sequenceGraph.getConnectedComponentOffset(sequenceBase, warpID + 1)};
            const auto connectedComponentSize{connectedComponentEnd - connectedComponentStart};

            // Get warp frontier
            const auto d_buffersBase{d_buffers + (connectedComponentStart << 1)};
            const DoubleBuffer warpDoubleBuffer{connectedComponentSize, d_buffersBase, d_buffersBase + connectedComponentSize, 0, false};
            Frontier warpFrontier{0, 0, warpDoubleBuffer, shared_isInQueue + warpIdx * connectedComponentSize * sizeof(bool), true};

            // Get chunked queue size and tail
            const auto chunkedQueueSize{warpFrontier.getMaxSize() >> ::__ffs(sizeof(queueChunk_t))};
            const auto chunkedQueueTail{warpFrontier.getMaxSize() & (sizeof(queueChunk_t) - 1)};

            // Perform insertions
            // Visit all neighbors of nodes in the current connected component (using stride access)
            for (auto connectedComponentIdx{connectedComponentStart + laneIdx}; connectedComponentIdx < connectedComponentEnd; connectedComponentIdx += KernelUtils::WARP_SIZE) {
                // Get node index
                const auto nodeIdx{d_sequenceGraph.getConnectedComponentMapping(sequenceBase, connectedComponentIdx)};

                // Get node cost in the current layer
                const auto currentLayerNodeCost{d_sequenceGraph.getCostsDoubleBuffer().current()[nodeIdx]};

                // Loop over all neighbors
                const auto neighborsStart{d_sequenceGraph.getPangenomeGraph().getNeighborsOffset(nodeIdx)};
                const auto neighborEnd{d_sequenceGraph.getPangenomeGraph().getNeighborsOffset(nodeIdx + 1)};
                for (auto neighborOffset{neighborsStart}; neighborOffset < neighborEnd; ++neighborOffset) {
                    // Get neighbor index
                    const auto neighborIdx{d_sequenceGraph.getPangenomeGraph().getNeighbor(neighborOffset)};

                    // Get updated current layer neighbor cost
                    const auto updatedCurrentLayerNeighborCost{currentLayerNodeCost + SequenceGraph::INSERTION_COST};

                    // Process neighbor
                    processNeighbor(d_sequenceGraph, &warpFrontier, neighborIdx, updatedCurrentLayerNeighborCost, laneIdx);
                }
            }

            // Swap frontier
            warpFrontier.swap();

            // Perform propagations
            while (!warpFrontier.isEmpty()) {
                // Empty frontier queue (using 128 bit / 16 bytes chunks) if necessary
                for (nodeSize_t queueChunkIdx{laneIdx}; queueChunkIdx < chunkedQueueSize; queueChunkIdx += KernelUtils::WARP_SIZE) {
                    reinterpret_cast<queueChunk_t*>(warpFrontier.getIsInQueue())[queueChunkIdx] = ::make_uint4(0, 0, 0, 0);
                }
                for (auto queueIdx{chunkedQueueTail}; queueIdx < warpFrontier.getMaxSize(); queueIdx += KernelUtils::WARP_SIZE) {
                    warpFrontier.getIsInQueue()[queueIdx] = false;
                }

                // Visit all neighbors of nodes in the current frontier (using stride access)
                for (nodeSize_t frontierIdx{laneIdx}; frontierIdx < warpFrontier.getSize(); frontierIdx += KernelUtils::WARP_SIZE) {
                    // Get node index
                    const auto nodeIdx{warpFrontier.getValue(frontierIdx)};

                    // Get node cost in the current layer
                    const auto currentLayerNodeCost{d_sequenceGraph.getCostsDoubleBuffer().current()[nodeIdx]};

                    // Loop over all neighbors
                    const auto neighborsStart{d_sequenceGraph.getPangenomeGraph().getNeighborsOffset(nodeIdx)};
                    const auto neighborEnd{d_sequenceGraph.getPangenomeGraph().getNeighborsOffset(nodeIdx + 1)};
                    for (auto neighborOffset{neighborsStart}; neighborOffset < neighborEnd; ++neighborOffset) {
                        // Get neighbor index
                        const auto neighborIdx{d_sequenceGraph.getPangenomeGraph().getNeighbor(neighborOffset)};

                        // Get updated current layer neighbor cost
                        const auto updatedCurrentLayerNeighborCost{currentLayerNodeCost + SequenceGraph::INSERTION_COST};

                        // Process neighbor
                        processNeighbor(d_sequenceGraph, &warpFrontier, neighborIdx, updatedCurrentLayerNeighborCost, laneIdx);
                    }
                }

                // Swap frontier
                warpFrontier.swap();
            }
        }
    }

    __global__ void SequenceGraphKernels::minCost(const SequenceGraph d_sequenceGraph, const scoreSize_t scoreIdx) { // NOLINT
        // Shared memory array to store the warp-level minima inside the block
        extern __shared__ cost_t shared_minCosts[];

        // Get thread id
        const auto threadId{::blockIdx.x * ::blockDim.x + ::threadIdx.x};

        // Initialize thread-level minimum
        auto threadMinCost{SequenceGraph::COST_MAX_VALUE};

        // Grid-level reduction using stride (safe for overflowing threads)
        const auto stride{::blockDim.x * ::gridDim.x};
        for (nodeSize_t nodeIdx{threadId}; nodeIdx < d_sequenceGraph.getPangenomeGraph().getNumNodes(); nodeIdx += stride) {
            const cost_t cost{d_sequenceGraph.getCostsDoubleBuffer()[nodeIdx]};
            threadMinCost = (cost < threadMinCost)? cost : threadMinCost;
        }

        // Warp-level reduction using warp shuffling (safe for overflowing threads)
        const auto warpIdx{::threadIdx.x >> KernelUtils::WARP_SHIFT};
        const auto laneIdx{::threadIdx.x & (KernelUtils::WARP_SIZE - 1)};
#pragma unroll
        for (auto offset{KernelUtils::WARP_SIZE / 2}; offset > 0; offset >>= 1) {
            const auto shuffledValue{::__shfl_down_sync(KernelUtils::BROADCAST_SHUFFLE_MASK, threadMinCost, offset)};
            threadMinCost = (shuffledValue < threadMinCost)? shuffledValue : threadMinCost;
        }

        // Store warp-level minimum using the first thread of each warp
        if (laneIdx == 0) {
            shared_minCosts[warpIdx] = threadMinCost;
        }
        ::__syncthreads();

        // Block-level reduction using the first warp of each block
        if (warpIdx == 0) {
            // Get warp-level minimum
            const auto numWarpsPerBlock{(::blockDim.x + KernelUtils::WARP_SIZE - 1) >> KernelUtils::WARP_SHIFT};
            threadMinCost = (::threadIdx.x < numWarpsPerBlock)? shared_minCosts[::threadIdx.x] : SequenceGraph::COST_MAX_VALUE;

            // Block-level reduction using warp shuffling
#pragma unroll
            for (auto offset{KernelUtils::WARP_SIZE / 2}; offset > 0; offset >>= 1) {
                const auto shuffledValue{::__shfl_down_sync(KernelUtils::BROADCAST_SHUFFLE_MASK, threadMinCost, offset)};
                threadMinCost = (shuffledValue < threadMinCost)? shuffledValue : threadMinCost;
            }

            // Final grid-level reduction using the first thread of each block
            if (laneIdx == 0) {
                ::atomicMin(&d_sequenceGraph.getScores()[scoreIdx], threadMinCost);
            }
        }
    }
} // cuSGA