#include "SequenceGraph.cuh"

#include <fstream>

#include "KernelUtils.cuh"

namespace cuSGA {
    __host__ SequenceGraph::SequenceGraph(const ::std::string& pangenomeGraphFileName, const ::std::string& sequenceFileName, ::std::string const (& connectedComponentsFileNames)[NUM_BASES], bool ownsInstance, SequenceGraph* pinned_instanceOptional, KernelUtils::BumpPtrAllocator* allocatorOptional) : SequenceGraph{PangenomeGraph{}, PackedDNASequence{}, {0}, {nullptr}, {nullptr}, {0}, DoubleBuffer<cost_t>{}, 0, nullptr, ownsInstance, pinned_instanceOptional} {
        // Get allocator
        KernelUtils::BumpPtrAllocator* allocator{allocatorOptional};
        KernelUtils::BumpPtrAllocator allocatorInstance{};
        if (!allocator) {
            allocator = &allocatorInstance;
        }

        // Open pangenome graph file
        auto pangenomeGraphFile{Utils::openFile(pangenomeGraphFileName)};

        // Read number of nodes from file
        nodeSize_t numNodes{0};
        if (!(pangenomeGraphFile >> numNodes)) {
            throw ::std::runtime_error{::std::format("An error occurred while reading values from file: {}", pangenomeGraphFileName)};
        }

        // Read number of edges from file
        edgeSize_t numEdges{0};
        if (!(pangenomeGraphFile >> numEdges)) {
            throw ::std::runtime_error{::std::format("An error occurred while reading values from file: {}", pangenomeGraphFileName)};
        }

        // Close pangenome graph file
        pangenomeGraphFile.close();

        // Open sequence file
        auto sequenceFile{Utils::openFile(sequenceFileName)};

        // Read number of scores from file
        scoreSize_t numScores{0};
        if (!(sequenceFile >> numScores)) {
            throw ::std::runtime_error{::std::format("An error occurred while reading values from file: {}", sequenceFileName)};
        }

        // Read maximum sequence length from file
        sequenceSize_t maxSequenceLength{0};
        if (!(sequenceFile >> maxSequenceLength)) {
            throw ::std::runtime_error{::std::format("An error occurred while reading values from file: {}", sequenceFileName)};
        }

        // Close sequence file
        sequenceFile.close();

        connectedComponentSize_t totalNumConnectedComponents{0};
#pragma unroll
        for (DNABase_t baseIdx{0}; baseIdx < NUM_BASES; ++baseIdx) {
            // Open connected components file
            auto connectedComponentsFile{Utils::openFile(connectedComponentsFileNames[baseIdx])};

            // Read number of connected components from file
            connectedComponentSize_t numConnectedComponents{0};
            if (!(connectedComponentsFile >> numConnectedComponents)) {
                throw ::std::runtime_error{::std::format("An error occurred while reading values from file: {}", connectedComponentsFileNames[baseIdx])};
            }

            // Accumulate number of connected components
            totalNumConnectedComponents += numConnectedComponents;

            // Close sequence file
            connectedComponentsFile.close();
        }

        // Grow allocator
        if (ownsInstance) {
            allocator->emplaceReserve<SequenceGraph>();
            growBuffers(allocator, numNodes, numEdges, maxSequenceLength, totalNumConnectedComponents, numScores);
        }

        // Initialize allocator
        if (ownsInstance) {
            allocator->initHostPinnedMem();
        }

        // Emplace buffers
        if (ownsInstance) {
            this->pinned_instance = allocator->emplaceReserve<SequenceGraph>();
        }
        this->pangenomeGraph = PangenomeGraph{pangenomeGraphFileName, false, &pinned_instance->pangenomeGraph, allocator};
        this->sequence = PackedDNASequence{sequenceFileName, false, &pinned_instance->sequence, allocator};
        // ...
        // TODO: read remaining attributes, kernel to find maxs or not?
        this->costsDoubleBuffer = DoubleBuffer{numNodes, false, &pinned_instance->costsDoubleBuffer, allocator};
        this->numScores = numScores;
        this->scores = allocator->emplaceSet<::std::remove_pointer_t<decltype(scores)>>(-1, numScores);

        // TODO
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
    }

    __host__ SequenceGraph SequenceGraph::copyToDevice(SequenceGraph* d_instanceOptional, KernelUtils::BumpPtrAllocator* allocatorOptional) {
        // Check if device instance already exists for this sequence graph
        if (d_instance) {
            throw ::std::runtime_error{"Device instance already exists for this Sequence Graph!"};
        }

        // Get allocator
        KernelUtils::BumpPtrAllocator* allocator{allocatorOptional};
        KernelUtils::BumpPtrAllocator allocatorInstance{};
        if (!allocator) {
            allocator = &allocatorInstance;
        }

        connectedComponentSize_t totalNumConnectedComponents{0};
#pragma unroll
        for (DNABase_t i{0}; i < NUM_BASES; ++i) {
            totalNumConnectedComponents += numConnectedComponents[i];
        }

        // Grow allocator
        if (ownsInstance) {
            allocator->grow<SequenceGraph>();
            growBuffersWithoutSequence(allocator, pangenomeGraph.getNumNodes(), pangenomeGraph.getNumEdges(), totalNumConnectedComponents, numScores);
        }

        // Initialize allocator
        if (ownsInstance) {
            allocator->initCudaGMem();
        }

        // Reserve instance
        if (ownsInstance) {
            this->d_instance = allocator->emplaceReserve<SequenceGraph>();
        }
        else {
            this->d_instance = d_instanceOptional;
        }

        // Emplace buffers
        const auto d_pangenomeGraph{pangenomeGraph.copyToDevice(&d_instance->pangenomeGraph, allocator)};
        const auto d_connectedComponentsOffsetsBase{allocator->cudaEmplaceCopy<::std::remove_pointer_t<::std::remove_reference_t<decltype(connectedComponentsOffsets[0])>>>(connectedComponentsOffsets[0], ::cudaMemcpyHostToDevice, totalNumConnectedComponents, false, cudaStreamDefault)};
        const auto d_connectedComponentsOffsetsMappingsBase{allocator->cudaEmplaceCopy<::std::remove_pointer_t<::std::remove_reference_t<decltype(connectedComponentsMappings[0])>>>(connectedComponentsMappings[0], ::cudaMemcpyHostToDevice, NUM_BASES * pangenomeGraph.getNumNodes(), false, cudaStreamDefault)};
        nodeSize_t* d_connectedComponentsOffsets[NUM_BASES]{nullptr};
        nodeSize_t* d_connectedComponentsMappings[NUM_BASES]{nullptr};
        connectedComponentSize_t connectedComponentsCounter{0};
#pragma unroll
        for (DNABase_t baseIdx{0}; baseIdx < NUM_BASES; ++baseIdx) {
            d_connectedComponentsOffsets[baseIdx] = d_connectedComponentsOffsetsBase + connectedComponentsCounter;
            d_connectedComponentsMappings[baseIdx] = d_connectedComponentsOffsetsMappingsBase + baseIdx * pangenomeGraph.getNumNodes();
            connectedComponentsCounter += numConnectedComponents[baseIdx];
        }
        const auto d_costsDoubleBuffer{costsDoubleBuffer.copyToDevice(&d_instance->costsDoubleBuffer, allocator)};
        const auto d_scores{allocator->cudaEmplaceCopy<::std::remove_pointer_t<decltype(scores)>>(scores, ::cudaMemcpyHostToDevice, numScores, false, cudaStreamDefault)};

        // Create temporary host instance holding the device pointers
        const SequenceGraph d_sequenceGraph{d_pangenomeGraph, sequence, numConnectedComponents, d_connectedComponentsOffsets, d_connectedComponentsMappings, maxConnectedComponentsSizes, d_costsDoubleBuffer, numScores, d_scores, ownsInstance, pinned_instance, d_instance};

        // Emplace instance
        if (ownsInstance) {
            *pinned_instance = d_sequenceGraph;
            CUDA_CHECK(::cudaMemcpyAsync(d_instance, pinned_instance, sizeof(SequenceGraph), ::cudaMemcpyHostToDevice, cudaStreamDefault));
        }

        return d_sequenceGraph;
    }

    __host__ void SequenceGraph::free() const {
        // Free device memory if present
        if (d_instance) {
            if (ownsInstance) {
                CUDA_CHECK(::cudaFreeAsync(d_instance, cudaStreamDefault));
            }
            else {
                CUDA_CHECK(::cudaFreeAsync(pinned_instance->getBuffersRoot(), cudaStreamDefault));
            }
        }

        // Free host memory
        if (ownsInstance) {
            CUDA_CHECK(::cudaFreeHost(pinned_instance));
        }
        else {
            CUDA_CHECK(::cudaFreeHost(getBuffersRoot()));
        }
    }

    __host__ cost_t* SequenceGraph::align(const ::std::string& sequenceFileName, nodeSize_t* const d_buffers) {
        // Open file
        ::std::ifstream sequenceFile{sequenceFileName};
        if (!sequenceFile.is_open()) {
            throw ::std::runtime_error{::std::format("Unable to open file: {}", sequenceFileName)};
        }

        // Copy sequence graph instance to device
        copyToDevice();

        // Loop over all sequences in the input file
        scoreSize_t scoreIdx{0};
        while (sequence.readFromFile(sequenceFileName, &sequenceFile)) {
            // Check for non-empty sequence
            if (sequence.getNumBases() == 0) {
                throw ::std::runtime_error{"Unable to align an empty sequence!"};
            }

            // Perform initialization step
            initialize();

            // Solve alignment layer by layer
            for (sequenceSize_t layerIdx{1}; layerIdx < sequence.getNumBases(); ++layerIdx) {
                // Perform deletions for the given layer
                deletions();

                // Perform substitutions for the given layer
                substitutions(layerIdx);

                // Skip insertions and propagations for the last layer
                if (layerIdx < sequence.getNumBases() - 1) {
                    // Perform insertions and propagations for the given layer
                    insertionsAndPropagations(layerIdx, d_buffers);

                    // Swap costs double buffer for the next layer
                    pinned_instance->costsDoubleBuffer.swap();
                }
            }

            // Compute minimum cost
            minCost(scoreIdx);

            // Move to the next score
            ++scoreIdx;
        }

        // Close file
        sequenceFile.close();

        // Copy over scores from device
        const auto scores{h2d_getScores()};

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
            d_sequenceGraph.getCostsDoubleBuffer().alternate()[nodeIdx] = initialNodeCost;
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
            // NOTE: Because deletions are run before propagations and used as an "initialization step" for layers different from the first, we don't need to use atomics
            d_sequenceGraph.getCostsDoubleBuffer().current()[nodeIdx] = currentLayerNodeCost;
        }
    }

    __device__ __forceinline__ void SequenceGraphKernels::processNeighbor(const SequenceGraph& d_sequenceGraph, Frontier* const warpFrontier, Frontier* const shared_frontier, const nodeSize_t neighborIdx, const cost_t updatedCurrentLayerNeighborCost, const ::uint8_t laneIdx) {
        // Set cost to atomic min and get previous current layer neighbor cost
        // NOTE: Because in-degree for a node should be low, we can avoid doing warp / block level reduction in order to reduce overhead
        const auto previousCurrentLayerNeighborCost{::atomicMin(&d_sequenceGraph.getCostsDoubleBuffer().current()[neighborIdx], updatedCurrentLayerNeighborCost)};

        // Get active mask
        const auto activeMask{::__activemask()};

        // Check for improvement
        bool wantsToInsert{(updatedCurrentLayerNeighborCost < previousCurrentLayerNeighborCost) && shared_frontier->isNodeInQueue(neighborIdx)};

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

        // Get number of insertions
        const auto numInsertions{::__popc(improvedMask)};
        const auto numInsertionsInSharedFrontier = ::min(numInsertions, SHARED_FRONTIER_BUFFER_SIZE - shared_frontier->getQueueSize());
        const auto numInsertionsInGlobalFrontier = numInsertions - numInsertionsInSharedFrontier;

        // Check if thread passed deduplication check
        if (wantsToInsert) {
            // Get insertion offset and check if inserting in shared or global frontier
            if (const auto insertionOffset{::__popc(improvedMask & ((1u << laneIdx) - 1))}; insertionOffset < numInsertionsInSharedFrontier) {
                // Insert node into next shared frontier
                shared_frontier->atomicInsertNodeInQueue(neighborIdx, shared_frontier->getQueueSize() + insertionOffset);
            }
            else {
                // Insert node into next global frontier
                warpFrontier->atomicInsertNodeInQueue(neighborIdx, warpFrontier->getQueueSize() + insertionOffset - numInsertionsInSharedFrontier);
            }
        }

        // Grow queue size
        shared_frontier->growQueueSize(numInsertionsInSharedFrontier);
        warpFrontier->growQueueSize(numInsertionsInGlobalFrontier);
    }

    __global__ void SequenceGraphKernels::insertionsAndPropagations(const SequenceGraph d_sequenceGraph, const DNABase sequenceBase, const targetSize_t numWarpsPerBlock, const connectedComponentSize_t maxConnectedComponentSize, nodeSize_t* const d_buffers) { // NOLINT
        // Shared memory, partitioned in the following way to guarantee alignment without wasting any space:
        //      |  buffers (nodeSize_t)  |  isInQueue (bool)  |
        extern __shared__ nodeSize_t shared_buffers[];
        const auto shared_isInQueue{shared_buffers + numWarpsPerBlock * SHARED_FRONTIER_BUFFER_SIZE};

        // Get thread warp ID and check for thread overflow
        if (const auto warpID{(::blockIdx.x * ::blockDim.x + ::threadIdx.x) >> KernelUtils::WARP_SHIFT}; warpID < d_sequenceGraph.getNumConnectedComponents(sequenceBase)) {
            // Get warp index and lane index
            const auto warpIdx{::threadIdx.x >> KernelUtils::WARP_SHIFT};
            const auto laneIdx{::threadIdx.x & (KernelUtils::WARP_SIZE - 1)};

            // Get connected component details
            const auto connectedComponentStart{d_sequenceGraph.getConnectedComponentOffset(sequenceBase, warpID)};
            const auto connectedComponentEnd{d_sequenceGraph.getConnectedComponentOffset(sequenceBase, warpID + 1)};
            const auto connectedComponentSize{connectedComponentEnd - connectedComponentStart};
            const auto packedQueueSize{(maxConnectedComponentSize + Frontier::PACKING_FACTOR - 1) >> Frontier::PACK_SHIFT};

            // Get shared frontier
            const auto shared_buffersBase{shared_buffers + ((warpIdx * SHARED_FRONTIER_BUFFER_SIZE) << 1)};
            const DoubleBuffer shared_doubleBuffer{SHARED_FRONTIER_BUFFER_SIZE, shared_buffersBase, shared_buffersBase + SHARED_FRONTIER_BUFFER_SIZE, 0, false};
            Frontier shared_frontier{0, 0, shared_doubleBuffer, shared_isInQueue + warpIdx * packedQueueSize, true};

            // Get warp frontier
            const auto d_buffersBase{d_buffers + (connectedComponentStart << 1)};
            const DoubleBuffer warpDoubleBuffer{connectedComponentSize, d_buffersBase, d_buffersBase + connectedComponentSize, 0, false};
            Frontier warpFrontier{0, 0, warpDoubleBuffer, shared_isInQueue + warpIdx * packedQueueSize, true};

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
                    processNeighbor(d_sequenceGraph, &warpFrontier, &shared_frontier, neighborIdx, updatedCurrentLayerNeighborCost, laneIdx);
                }
            }

            // Swap frontier
            shared_frontier.swap();
            warpFrontier.swap();

            // Perform propagations
            while (!(shared_frontier.isEmpty() && warpFrontier.isEmpty())) {
                // Empty frontier queue if necessary
                for (nodeSize_t queuePackIdx{laneIdx}; queuePackIdx < packedQueueSize; queuePackIdx += KernelUtils::WARP_SIZE) {
                    warpFrontier.getIsInQueue()[queuePackIdx] = 0;
                }

                // Visit all neighbors of nodes in the current shared frontier (using stride access)
                for (nodeSize_t frontierIdx{laneIdx}; frontierIdx < shared_frontier.getSize(); frontierIdx += KernelUtils::WARP_SIZE) {
                    // Get node index
                    const auto nodeIdx{shared_frontier.getValue(frontierIdx)};

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
                        processNeighbor(d_sequenceGraph, &warpFrontier, &shared_frontier, neighborIdx, updatedCurrentLayerNeighborCost, laneIdx);
                    }
                }

                // Visit all neighbors of nodes in the current global frontier (using stride access)
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
                        processNeighbor(d_sequenceGraph, &warpFrontier, &shared_frontier, neighborIdx, updatedCurrentLayerNeighborCost, laneIdx);
                    }
                }

                // Swap frontier
                shared_frontier.swap();
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
            const cost_t cost{d_sequenceGraph.getCostsDoubleBuffer().current()[nodeIdx]};
            threadMinCost = ::min(cost, threadMinCost);
        }

        // Warp-level reduction using warp shuffling (safe for overflowing threads)
        const auto warpIdx{::threadIdx.x >> KernelUtils::WARP_SHIFT};
        const auto laneIdx{::threadIdx.x & (KernelUtils::WARP_SIZE - 1)};
#pragma unroll
        for (auto offset{KernelUtils::WARP_SIZE / 2}; offset > 0; offset >>= 1) {
            const auto shuffledValue{::__shfl_down_sync(KernelUtils::BROADCAST_SHUFFLE_MASK, threadMinCost, offset)};
            threadMinCost = ::min(shuffledValue, threadMinCost);
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
                threadMinCost = ::min(shuffledValue, threadMinCost);
            }

            // Final grid-level reduction using the first thread of each block
            if (laneIdx == 0) {
                ::atomicMin(&d_sequenceGraph.getScores()[scoreIdx], threadMinCost);
            }
        }
    }
} // cuSGA