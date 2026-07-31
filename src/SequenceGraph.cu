#include "SequenceGraph.cuh"

#include <fstream>
#include <iostream>

#include "KernelUtils.cuh"

namespace cuSGA {
    __host__ SequenceGraph::SequenceGraph(const ::std::string& pangenomeGraphFileName, const ::std::string& sequenceFileName, ::std::string const (& connectedComponentsFileNames)[NUM_BASES], bool ownsInstance, SequenceGraph* __restrict__ pinned_instanceOptional, KernelUtils::BumpPtrAllocator* allocatorOptional) : SequenceGraph{PangenomeGraph{}, PackedDNASequence{}, {0}, {nullptr}, {nullptr}, {nullptr}, {nullptr}, {0}, DoubleBuffer<cost_t>{}, 0, nullptr, ownsInstance, pinned_instanceOptional} {
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

        // Read total number of connected components from files
        ::std::ifstream connectedComponentsFiles[NUM_BASES]{};
        connectedComponentSize_t totalNumConnectedComponents{0};
        connectedComponentSize_t numConnectedComponents[NUM_BASES]{};
#pragma unroll
        for (DNABase_t baseIdx{0}; baseIdx < NUM_BASES; ++baseIdx) {
            // Open connected components file
            connectedComponentsFiles[baseIdx] = Utils::openFile(connectedComponentsFileNames[baseIdx]);

            // Read number of connected components from file
            if (!(connectedComponentsFiles[baseIdx] >> numConnectedComponents[baseIdx])) {
                throw ::std::runtime_error{::std::format("An error occurred while reading values from file: {}", connectedComponentsFileNames[baseIdx])};
            }

            // Accumulate number of connected components
            totalNumConnectedComponents += numConnectedComponents[baseIdx];
        }

        // Grow allocator
        if (ownsInstance) {
            allocator->grow<SequenceGraph>();
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
        const auto connectedComponentsOffsetsBase{allocator->emplaceReserve<::std::remove_pointer_t<::std::remove_reference_t<decltype(connectedComponentsOffsets[0])>>>(totalNumConnectedComponents + NUM_BASES)};
        const auto connectedComponentsMappingsBase{allocator->emplaceReserve<::std::remove_pointer_t<::std::remove_reference_t<decltype(connectedComponentsMappings[0])>>>(NUM_BASES * numNodes)};
        const auto connectedComponentsReverseMappingsBase{allocator->emplaceReserve<::std::remove_pointer_t<::std::remove_reference_t<decltype(connectedComponentsReverseMappings[0])>>>(NUM_BASES * numNodes)};
        const auto connectedComponentsLocalIndexMappingsBase{allocator->emplaceReserve<::std::remove_pointer_t<::std::remove_reference_t<decltype(connectedComponentsLocalIndexMappings[0])>>>(NUM_BASES * numNodes)};
        connectedComponentSize_t connectedComponentsCounter{0};
#pragma unroll
        for (DNABase_t baseIdx{0}; baseIdx < NUM_BASES; ++baseIdx) {
            // Get connected components file
            auto connectedComponentsFile{std::move(connectedComponentsFiles[baseIdx])};

            // Set number of connected components
            const auto numConnectedComponentsValue{numConnectedComponents[baseIdx]};
            this->numConnectedComponents[baseIdx] = numConnectedComponentsValue;

            // Read connected components offsets from file
            const auto connectedComponentsOffsets{connectedComponentsOffsetsBase + connectedComponentsCounter};
            for (connectedComponentSize_t connectedComponentIdx{0}; connectedComponentIdx <= numConnectedComponentsValue; ++connectedComponentIdx) {
                if (!(connectedComponentsFile >> connectedComponentsOffsets[connectedComponentIdx])) {
                    throw ::std::runtime_error{::std::format("An error occurred while reading values from file: {}", connectedComponentsFileNames[baseIdx])};
                }
            }
            this->connectedComponentsOffsets[baseIdx] = connectedComponentsOffsets;

            // Read connected components mappings from file and build inverse local index map
            const auto connectedComponentsMappings{connectedComponentsMappingsBase + baseIdx * numNodes};
            const auto connectedComponentsLocalIndexMappings{connectedComponentsLocalIndexMappingsBase + baseIdx * numNodes};
            for (connectedComponentSize_t connectedComponentIndex{0}; connectedComponentIndex < numConnectedComponentsValue; ++connectedComponentIndex) {
                const auto connectedComponentStart{connectedComponentsOffsets[connectedComponentIndex]};
                const auto connectedComponentEnd{connectedComponentsOffsets[connectedComponentIndex + 1]};
                for (nodeSize_t nodeIdx{connectedComponentStart}; nodeIdx < connectedComponentEnd; ++nodeIdx) {
                    nodeSize_t globalNodeIdx{0};
                    if (!(connectedComponentsFile >> globalNodeIdx)) {
                        throw ::std::runtime_error{::std::format("An error occurred while reading values from file: {}", connectedComponentsFileNames[baseIdx])};
                    }
                    connectedComponentsMappings[nodeIdx] = globalNodeIdx;
                    connectedComponentsLocalIndexMappings[globalNodeIdx] = nodeIdx - connectedComponentStart;
                }
            }
            this->connectedComponentsMappings[baseIdx] = connectedComponentsMappings;
            this->connectedComponentsLocalIndexMappings[baseIdx] = connectedComponentsLocalIndexMappings;

            // Read connected components reverse mappings from file
            const auto connectedComponentsReverseMappings{connectedComponentsReverseMappingsBase + baseIdx * numNodes};
            for (nodeSize_t nodeIdx{0}; nodeIdx < numNodes; ++nodeIdx) {
                connectedComponentSize_t connectedComponentReverseMapping{0};
                if (!(connectedComponentsFile >> connectedComponentReverseMapping)) {
                    throw ::std::runtime_error{::std::format("An error occurred while reading values from file: {}", connectedComponentsFileNames[baseIdx])};
                }
                connectedComponentsReverseMappings[nodeIdx] = connectedComponentReverseMapping;
            }
            this->connectedComponentsReverseMappings[baseIdx] = connectedComponentsReverseMappings;

            // Find max connected component size for this base
            connectedComponentSize_t maxConnectedComponentsSize{0};
            for (connectedComponentSize_t connectedComponentIdx{0}; connectedComponentIdx < numConnectedComponentsValue; ++connectedComponentIdx) {
                const auto currentConnectedComponentSize{connectedComponentsOffsets[connectedComponentIdx + 1] - connectedComponentsOffsets[connectedComponentIdx]};
                maxConnectedComponentsSize = ::std::max(maxConnectedComponentsSize, currentConnectedComponentSize);
            }
            this->maxConnectedComponentsSizes[baseIdx] = maxConnectedComponentsSize;

            // Increase number of connected components counter
            connectedComponentsCounter += (numConnectedComponents[baseIdx] + 1);

            // Close connected components file
            connectedComponentsFile.close();
        }
        this->costsDoubleBuffer = DoubleBuffer{numNodes, false, &pinned_instance->costsDoubleBuffer, allocator};
        this->numScores = numScores;
        this->scores = allocator->emplaceSet<::std::remove_pointer_t<decltype(scores)>>(-1, numScores);

        // Close pangenome graph file
        pangenomeGraphFile.close();

        // Close sequence file
        sequenceFile.close();
    }

    __host__ SequenceGraph SequenceGraph::copyToDevice(SequenceGraph* __restrict__ d_instanceOptional, KernelUtils::BumpPtrAllocator* allocatorOptional) {
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
        const auto d_connectedComponentsOffsetsBase{allocator->cudaEmplaceCopy<::std::remove_pointer_t<::std::remove_reference_t<decltype(connectedComponentsOffsets[0])>>>(connectedComponentsOffsets[0], ::cudaMemcpyHostToDevice, totalNumConnectedComponents + NUM_BASES, false, cudaStreamDefault)};
        const auto d_connectedComponentsMappingsBase{allocator->cudaEmplaceCopy<::std::remove_pointer_t<::std::remove_reference_t<decltype(connectedComponentsMappings[0])>>>(connectedComponentsMappings[0], ::cudaMemcpyHostToDevice, NUM_BASES * pangenomeGraph.getNumNodes(), false, cudaStreamDefault)};
        const auto d_connectedComponentsReverseMappingsBase{allocator->cudaEmplaceCopy<::std::remove_pointer_t<::std::remove_reference_t<decltype(connectedComponentsReverseMappings[0])>>>(connectedComponentsReverseMappings[0], ::cudaMemcpyHostToDevice, NUM_BASES * pangenomeGraph.getNumNodes(), false, cudaStreamDefault)};
        const auto d_connectedComponentsLocalIndexMappingsBase{allocator->cudaEmplaceCopy<::std::remove_pointer_t<::std::remove_reference_t<decltype(connectedComponentsLocalIndexMappings[0])>>>(connectedComponentsLocalIndexMappings[0], ::cudaMemcpyHostToDevice, NUM_BASES * pangenomeGraph.getNumNodes(), false, cudaStreamDefault)};
        nodeSize_t* d_connectedComponentsOffsets[NUM_BASES]{nullptr};
        nodeSize_t* d_connectedComponentsMappings[NUM_BASES]{nullptr};
        connectedComponentSize_t* d_connectedComponentsReverseMappings[NUM_BASES]{nullptr};
        connectedComponentSize_t* d_connectedComponentsLocalIndexMappings[NUM_BASES]{nullptr};
        connectedComponentSize_t connectedComponentsCounter{0};
#pragma unroll
        for (DNABase_t baseIdx{0}; baseIdx < NUM_BASES; ++baseIdx) {
            d_connectedComponentsOffsets[baseIdx] = d_connectedComponentsOffsetsBase + connectedComponentsCounter;
            d_connectedComponentsMappings[baseIdx] = d_connectedComponentsMappingsBase + baseIdx * pangenomeGraph.getNumNodes();
            d_connectedComponentsReverseMappings[baseIdx] = d_connectedComponentsReverseMappingsBase + baseIdx * pangenomeGraph.getNumNodes();
            d_connectedComponentsLocalIndexMappings[baseIdx] = d_connectedComponentsLocalIndexMappingsBase + baseIdx * pangenomeGraph.getNumNodes();
            connectedComponentsCounter += (numConnectedComponents[baseIdx] + 1);
        }
        const auto d_costsDoubleBuffer{costsDoubleBuffer.copyToDevice(&d_instance->costsDoubleBuffer, allocator)};
        const auto d_scores{allocator->cudaEmplaceCopy<::std::remove_pointer_t<decltype(scores)>>(scores, ::cudaMemcpyHostToDevice, numScores, false, cudaStreamDefault)};

        // Create temporary host instance holding the device pointers
        const SequenceGraph d_sequenceGraph{d_pangenomeGraph, sequence, numConnectedComponents, d_connectedComponentsOffsets, d_connectedComponentsMappings, d_connectedComponentsReverseMappings, d_connectedComponentsLocalIndexMappings, maxConnectedComponentsSizes, d_costsDoubleBuffer, numScores, d_scores, ownsInstance, pinned_instance, d_instance};

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

    __host__ cost_t* SequenceGraph::connectedComponentsAlign(const ::std::string& sequenceFileName) {
        // Open file
        ::std::ifstream sequenceFile{sequenceFileName};
        if (!sequenceFile.is_open()) {
            throw ::std::runtime_error{::std::format("Unable to open file: {}", sequenceFileName)};
        }

        // Read and skip number of scores and max sequence length
        if (scoreSize_t numScores{0}; !(sequenceFile >> numScores)) {
            throw ::std::runtime_error{::std::format("An error occurred while reading values from file: {}", sequenceFileName)};
        }
        if (sequenceSize_t maxSequenceLength{0}; !(sequenceFile >> maxSequenceLength)) {
            throw ::std::runtime_error{::std::format("An error occurred while reading values from file: {}", sequenceFileName)};
        }

        // Copy sequence graph instance to device
        copyToDevice();

        // Allocate additional device buffers
        connectedComponentSize_t maxNumConnectedComponents{0};
#pragma unroll
        for (DNABase_t baseIdx{0}; baseIdx < NUM_BASES; ++baseIdx) {
            maxNumConnectedComponents = ::std::max(maxNumConnectedComponents, numConnectedComponents[baseIdx]);
        }
        nodeSize_t* d_buffers{nullptr};
        CUDA_CHECK(::cudaMallocAsync(&d_buffers, DoubleBuffer<nodeSize_t>::NUM_DOUBLE_BUFFERS * pangenomeGraph.getNumNodes() * sizeof(nodeSize_t), cudaStreamDefault));
        bool* d_needsVisiting{nullptr};
        CUDA_CHECK(::cudaMallocAsync(&d_needsVisiting, maxNumConnectedComponents * sizeof(bool), cudaStreamDefault));
        CUDA_CHECK(::cudaMemsetAsync(d_needsVisiting, 0, maxNumConnectedComponents * sizeof(bool), cudaStreamDefault));

        // Loop over all sequences in the input file
        scoreSize_t scoreIdx{0};
        while (sequence.readFromFile(sequenceFileName, &sequenceFile)) {
            // Check for non-empty sequence
            if (sequence.getNumBases() == 0) {
                throw ::std::runtime_error{"Unable to align an empty sequence!"};
            }

            // Perform initialization step
            initialize();

            // Swap costs double buffer for the next layer
            pinned_instance->costsDoubleBuffer.swap();

            // Solve alignment layer by layer
            const auto numBases{sequence.getNumBases()};
            for (sequenceSize_t layerIdx{1}; layerIdx < numBases; ++layerIdx) {
                // Perform deletions for the given layer
                deletions();

                // Perform substitutions for the given layer
                substitutions(layerIdx, d_needsVisiting);

                // Perform insertions and propagations for the given layer (early return if last layer)
                insertionsAndPropagations(layerIdx, d_buffers, d_needsVisiting);

                // Swap costs double buffer for the next layer (unless last layer)
                if (layerIdx < sequence.getNumBases() - 1) {
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

        // Free additional device buffers
        ::cudaFreeAsync(d_buffers, cudaStreamDefault);
        ::cudaFreeAsync(d_needsVisiting, cudaStreamDefault);

        return scores;
    }

    __host__ cost_t* SequenceGraph::gridAlign(const ::std::string& sequenceFileName) {
        // Open file
        ::std::ifstream sequenceFile{sequenceFileName};
        if (!sequenceFile.is_open()) {
            throw ::std::runtime_error{::std::format("Unable to open file: {}", sequenceFileName)};
        }

        // Read and skip number of scores and max sequence length
        if (scoreSize_t numScores{0}; !(sequenceFile >> numScores)) {
            throw ::std::runtime_error{::std::format("An error occurred while reading values from file: {}", sequenceFileName)};
        }
        if (sequenceSize_t maxSequenceLength{0}; !(sequenceFile >> maxSequenceLength)) {
            throw ::std::runtime_error{::std::format("An error occurred while reading values from file: {}", sequenceFileName)};
        }

        // Copy sequence graph instance to device
        copyToDevice();

        // Allocate additional device buffers
        nodeSize_t* d_frontierBufferSize1{nullptr};
        CUDA_CHECK(::cudaMallocAsync(&d_frontierBufferSize1, (2 + DoubleBuffer<nodeSize_t>::NUM_DOUBLE_BUFFERS * pangenomeGraph.getNumNodes()) * sizeof(nodeSize_t) + 2 * pangenomeGraph.getNumNodes() * sizeof(int), cudaStreamDefault));
        const auto d_frontierBufferSize2{d_frontierBufferSize1 + 1};
        const auto d_frontierBuffer1{d_frontierBufferSize2 + 1};
        const auto d_frontierBuffer2{d_frontierBuffer1 + pangenomeGraph.getNumNodes()};
        const auto d_frontierQueue1{reinterpret_cast<int*>(d_frontierBuffer2 + pangenomeGraph.getNumNodes())};
        const auto d_frontierQueue2{reinterpret_cast<int*>(d_frontierBuffer2 + pangenomeGraph.getNumNodes())};
        CUDA_CHECK(::cudaMemsetAsync(d_frontierQueue1, 0, 2 * pangenomeGraph.getNumNodes() * sizeof(int), cudaStreamDefault));

        // Loop over all sequences in the input file
        scoreSize_t scoreIdx{0};
        while (sequence.readFromFile(sequenceFileName, &sequenceFile)) {
            // Check for non-empty sequence
            if (sequence.getNumBases() == 0) {
                throw ::std::runtime_error{"Unable to align an empty sequence!"};
            }

            // Perform initialization step
            initialize();

            // Swap costs double buffer for the next layer
            pinned_instance->costsDoubleBuffer.swap();

            // Solve alignment layer by layer
            const auto numBases{sequence.getNumBases()};
            for (sequenceSize_t layerIdx{1}; layerIdx < numBases; ++layerIdx) {
                // Perform deletions for the given layer
                deletions();

                // Perform substitutions for the given layer
                cooperativeSubstitutions(layerIdx, d_frontierBufferSize1, d_frontierBuffer1, d_frontierQueue1);

                // Perform insertions and propagations for the given layer (early return if last layer)
                cooperativeInsertionsAndPropagations(layerIdx, d_frontierBuffer1, d_frontierBuffer2, d_frontierQueue1, d_frontierQueue2, d_frontierBufferSize1, d_frontierBufferSize2);

                // Swap costs double buffer for the next layer (unless last layer)
                if (layerIdx < sequence.getNumBases() - 1) {
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

        // Free additional device buffers
        ::cudaFreeAsync(d_frontierBufferSize1, cudaStreamDefault);

        return scores;
    }

    __host__ cost_t* SequenceGraph::gridBlockAggregationAlign(const ::std::string& sequenceFileName) {
        // Open file
        ::std::ifstream sequenceFile{sequenceFileName};
        if (!sequenceFile.is_open()) {
            throw ::std::runtime_error{::std::format("Unable to open file: {}", sequenceFileName)};
        }

        // Read and skip number of scores and max sequence length
        if (scoreSize_t numScores{0}; !(sequenceFile >> numScores)) {
            throw ::std::runtime_error{::std::format("An error occurred while reading values from file: {}", sequenceFileName)};
        }
        if (sequenceSize_t maxSequenceLength{0}; !(sequenceFile >> maxSequenceLength)) {
            throw ::std::runtime_error{::std::format("An error occurred while reading values from file: {}", sequenceFileName)};
        }

        // Copy sequence graph instance to device
        copyToDevice();

        // Allocate additional device buffers
        nodeSize_t* d_frontierBufferSize1{nullptr};
        CUDA_CHECK(::cudaMallocAsync(&d_frontierBufferSize1, (2 + DoubleBuffer<nodeSize_t>::NUM_DOUBLE_BUFFERS * pangenomeGraph.getNumNodes()) * sizeof(nodeSize_t) + 2 * pangenomeGraph.getNumNodes() * sizeof(int), cudaStreamDefault));
        const auto d_frontierBufferSize2{d_frontierBufferSize1 + 1};
        const auto d_frontierBuffer1{d_frontierBufferSize2 + 1};
        const auto d_frontierBuffer2{d_frontierBuffer1 + pangenomeGraph.getNumNodes()};
        const auto d_frontierQueue1{reinterpret_cast<int*>(d_frontierBuffer2 + pangenomeGraph.getNumNodes())};
        const auto d_frontierQueue2{reinterpret_cast<int*>(d_frontierBuffer2 + pangenomeGraph.getNumNodes())};
        CUDA_CHECK(::cudaMemsetAsync(d_frontierQueue1, 0, 2 * pangenomeGraph.getNumNodes() * sizeof(int), cudaStreamDefault));

        // Loop over all sequences in the input file
        scoreSize_t scoreIdx{0};
        while (sequence.readFromFile(sequenceFileName, &sequenceFile)) {
            // Check for non-empty sequence
            if (sequence.getNumBases() == 0) {
                throw ::std::runtime_error{"Unable to align an empty sequence!"};
            }

            // Perform initialization step
            initialize();

            // Swap costs double buffer for the next layer
            pinned_instance->costsDoubleBuffer.swap();

            // Solve alignment layer by layer
            const auto numBases{sequence.getNumBases()};
            for (sequenceSize_t layerIdx{1}; layerIdx < numBases; ++layerIdx) {
                // Perform deletions for the given layer
                deletions();

                // Perform substitutions for the given layer
                cooperativeSubstitutions(layerIdx, d_frontierBufferSize1, d_frontierBuffer1, d_frontierQueue1);

                // Perform insertions and propagations for the given layer (early return if last layer)
                cooperativeBlockAggregationInsertionsAndPropagations(layerIdx, d_frontierBuffer1, d_frontierBuffer2, d_frontierQueue1, d_frontierQueue2, d_frontierBufferSize1, d_frontierBufferSize2);

                // Swap costs double buffer for the next layer (unless last layer)
                if (layerIdx < sequence.getNumBases() - 1) {
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

        // Free additional device buffers
        ::cudaFreeAsync(d_frontierBufferSize1, cudaStreamDefault);

        return scores;
    }

    __global__ void SequenceGraphKernels::initialize(const sequencePack_t* __restrict__ const d_baseValues, cost_t* __restrict__ const d_currentCosts, const nodeSize_t numNodes, const DNABase initialSequenceBase) {
        // Get thread node index and check for thread overflow
        const auto nodeIdx{::blockIdx.x * ::blockDim.x + ::threadIdx.x};
        if (nodeIdx >= numNodes) {
            return;
        }

        // Get DNA base for the current node
        const auto nodeBase{PackedDNASequence::getBase(d_baseValues, nodeIdx)};

        // Compute updated node cost
        const auto initialNodeCost{(initialSequenceBase == nodeBase)? 0 : SequenceGraph::INITIALIZATION_COST};

        // Initialize node cost for the next layer
        d_currentCosts[nodeIdx] = initialNodeCost;
    }

    __global__ void SequenceGraphKernels::substitutions(const edgeSize_t* __restrict__ const d_neighborOffsets, const nodeSize_t* __restrict__ const d_neighborValues, const sequencePack_t* __restrict__ const d_baseValues, const cost_t* __restrict__ const d_previousCosts, cost_t* __restrict__ const d_currentCosts, bool* __restrict__ const d_needsVisiting, const connectedComponentSize_t* __restrict__ const connectedComponentsReverseMapping, const nodeSize_t numNodes, const DNABase sequenceBase) {
        // Get thread node index and check for thread overflow
        const auto nodeIdx{::blockIdx.x * ::blockDim.x + ::threadIdx.x};
        if (nodeIdx >= numNodes) {
            return;
        }

        // Get node cost in the previous layer
        const auto previousLayerNodeCost{d_previousCosts[nodeIdx]};

        // Loop over all neighbors
        const auto neighborsStart{d_neighborOffsets[nodeIdx]};
        const auto neighborEnd{d_neighborOffsets[nodeIdx + 1]};
        for (auto neighborOffset{neighborsStart}; neighborOffset < neighborEnd; ++neighborOffset) {
            // Get neighbor index
            const auto neighborIdx{d_neighborValues[neighborOffset]};

            // Get DNA base for the neighbor
            const auto neighborBase{PackedDNASequence::getBase(d_baseValues, neighborIdx)};

            // Compute updated node cost
            const auto updatedCurrentLayerNeighborCost{(sequenceBase == neighborBase)? previousLayerNodeCost : (previousLayerNodeCost + SequenceGraph::SUBSTITUTION_COST)};

            // Set cost to atomic min
            // NOTE: Because in-degree for a node should be low, we can avoid doing warp / block level reduction in order to reduce overhead
            if (const auto previousCurrentLayerNeighborCost{::atomicMin(d_currentCosts + neighborIdx, updatedCurrentLayerNeighborCost)}; updatedCurrentLayerNeighborCost < previousCurrentLayerNeighborCost) {
                const auto connectedComponentNeighborIdx{connectedComponentsReverseMapping[neighborIdx]};
                d_needsVisiting[connectedComponentNeighborIdx] = true;
            }
        }
    }

    __global__ void SequenceGraphKernels::cooperativeSubstitutions(const edgeSize_t* __restrict__ const d_neighborOffsets, const nodeSize_t* __restrict__ const d_neighborValues, const sequencePack_t* __restrict__ const d_baseValues, const cost_t* __restrict__ const d_previousCosts, cost_t* __restrict__ const d_currentCosts, nodeSize_t* __restrict__ const d_frontierBufferSize, nodeSize_t* __restrict__ const d_frontierBuffer, int* __restrict__ const d_frontierQueue, const nodeSize_t numNodes, const DNABase sequenceBase) {
        // Get thread node index and check for thread overflow
        const auto nodeIdx{::blockIdx.x * ::blockDim.x + ::threadIdx.x};
        if (nodeIdx >= numNodes) {
            return;
        }

        // Get node cost in the previous layer
        const auto previousLayerNodeCost{d_previousCosts[nodeIdx]};

        // Loop over all neighbors
        const auto neighborsStart{d_neighborOffsets[nodeIdx]};
        const auto neighborEnd{d_neighborOffsets[nodeIdx + 1]};
        for (auto neighborOffset{neighborsStart}; neighborOffset < neighborEnd; ++neighborOffset) {
            // Get neighbor index
            const auto neighborIdx{d_neighborValues[neighborOffset]};

            // Get DNA base for the neighbor
            const auto neighborBase{PackedDNASequence::getBase(d_baseValues, neighborIdx)};

            // Compute updated node cost
            const auto updatedCurrentLayerNeighborCost{(sequenceBase == neighborBase)? previousLayerNodeCost : (previousLayerNodeCost + SequenceGraph::SUBSTITUTION_COST)};

            // Set cost to atomic min
            if (const auto previousCurrentLayerNeighborCost{::atomicMin(d_currentCosts + neighborIdx, updatedCurrentLayerNeighborCost)}; (updatedCurrentLayerNeighborCost < previousCurrentLayerNeighborCost) && (!::atomicExch(d_frontierQueue + neighborIdx, 1))) {
                // Get active mask
                const auto activeMask{::__activemask()};

                // Get thread rank
                const auto threadRank{::__popc(activeMask & ((1u << threadIdx.x) - 1))};

                // Get number of insertions
                const auto numInsertions{::__popc(activeMask)};

                // Get leader lane index
                const auto leaderLaneIdx{::__ffs(static_cast<int>(activeMask)) - 1};

                // Get insertion base
                nodeSize_t insertionBase{0};

                // Leader thread reserves space for all threads in the warp
                if (::threadIdx.x == leaderLaneIdx) {
                    insertionBase = ::atomicAdd(d_frontierBufferSize, numInsertions);
                }

                // Shuffle insertion base
                insertionBase = ::__shfl_sync(activeMask, insertionBase, leaderLaneIdx);

                // Insert in queue
                d_frontierBuffer[insertionBase + threadRank] = neighborIdx;
            }
        }
    }

    __global__ void SequenceGraphKernels::deletions(const cost_t* __restrict__ const d_previousCosts, cost_t* __restrict__ const d_currentCosts, const nodeSize_t numNodes) {
        // Get thread node index and check for thread overflow
        const auto nodeIdx{::blockIdx.x * ::blockDim.x + ::threadIdx.x};
        if (nodeIdx >= numNodes) {
            return;
        }

        // Get node cost in the previous layer
        const auto previousLayerNodeCost{d_previousCosts[nodeIdx]};

        // Compute updated node cost
        const auto currentLayerNodeCost{previousLayerNodeCost + SequenceGraph::DELETION_COST};

        // Initialize node cost for the next layer
        // NOTE: Because deletions are run before propagations and used as an "initialization step" for layers different from the first, we don't need to use atomics
        d_currentCosts[nodeIdx] = currentLayerNodeCost;
    }

    __global__ void SequenceGraphKernels::insertionsAndPropagations(const edgeSize_t* __restrict__ const d_neighborOffsets, const nodeSize_t* __restrict__ const d_neighborValues, const nodeSize_t* __restrict__ const d_connectedComponentOffsets, const nodeSize_t* __restrict__ const d_connectedComponentMappings, const connectedComponentSize_t* __restrict__ const d_connectedComponentLocalIndexMappings, cost_t* __restrict__ const d_currentCosts, nodeSize_t* __restrict__ const d_buffers, bool* __restrict__ const d_needsVisiting, const targetSize_t numWarpsPerBlock, const connectedComponentSize_t numConnectedComponents, const connectedComponentSize_t maxConnectedComponentSize, const bool earlyExit) {
        // Shared memory, partitioned in the following way to guarantee alignment without wasting any space:
        //      |  buffers (nodeSize_t)  |  isInQueue (bool)  |
        extern __shared__ nodeSize_t shared_buffers[];
        const auto shared_isInQueue{reinterpret_cast<queuePack_t*>(shared_buffers + numWarpsPerBlock * (SHARED_FRONTIER_BUFFER_SIZE << 1))};

        // Get thread warp ID and check for thread overflow
        const auto warpID{(::blockIdx.x * ::blockDim.x + ::threadIdx.x) >> KernelUtils::WARP_SHIFT};
        if (warpID >= numConnectedComponents) {
            return;
        }

        // Check if connected component needs visiting
        if (const auto needsVisiting{d_needsVisiting[warpID]}; !needsVisiting) {
            return;
        }

        // Get connected component data
        const auto connectedComponentStart{d_connectedComponentOffsets[warpID]};
        const auto connectedComponentEnd{d_connectedComponentOffsets[warpID + 1]};
        const auto connectedComponentSize{connectedComponentEnd - connectedComponentStart};
        const auto packedQueueSize{(maxConnectedComponentSize + Frontier::PACKING_FACTOR - 1) >> Frontier::PACK_SHIFT};

        // Get warp index
        const auto warpIdx{::threadIdx.x >> KernelUtils::WARP_SHIFT};

        // Get warp frontier
        const auto d_buffersBase{d_buffers + (connectedComponentStart << 1)};
        const DoubleBuffer warpDoubleBuffer{connectedComponentSize, d_buffersBase, d_buffersBase + connectedComponentSize, 0, false};
        const auto shared_isInQueueBase{shared_isInQueue + warpIdx * packedQueueSize};
        Frontier warpFrontier{0, 0, warpDoubleBuffer, shared_isInQueueBase};

        // Get shared frontier
        const auto shared_buffersBase{shared_buffers + ((warpIdx * SHARED_FRONTIER_BUFFER_SIZE) << 1)};
        const DoubleBuffer shared_doubleBuffer{SHARED_FRONTIER_BUFFER_SIZE, shared_buffersBase, shared_buffersBase + SHARED_FRONTIER_BUFFER_SIZE, 0, false};
        Frontier shared_frontier{0, 0, shared_doubleBuffer, shared_isInQueueBase};

        // Get lane index
        const auto laneIdx{::threadIdx.x & (KernelUtils::WARP_SIZE - 1)};

        // Clear needs visiting flag
        if (laneIdx == 0) {
            d_needsVisiting[warpID] = false;
        }

        // Return early if early exit
        if (earlyExit) {
            return;
        }

        // Perform insertions
        // Visit all neighbors of nodes in the current connected component (using stride access)
        for (auto connectedComponentIdx{connectedComponentStart + laneIdx}; connectedComponentIdx < connectedComponentEnd; connectedComponentIdx += KernelUtils::WARP_SIZE) {
            // Get node index
            const auto nodeIdx{d_connectedComponentMappings[connectedComponentIdx]};

            // Process node
            processNode(nodeIdx, &warpFrontier, &shared_frontier, d_neighborOffsets, d_neighborValues, d_connectedComponentLocalIndexMappings, d_currentCosts);
        }

        // Sync frontier sizes
        shared_frontier.setQueueSize(::__shfl_sync(KernelUtils::BROADCAST_SHUFFLE_MASK, shared_frontier.getQueueSize(), 0));
        warpFrontier.setQueueSize(::__shfl_sync(KernelUtils::BROADCAST_SHUFFLE_MASK, warpFrontier.getQueueSize(), 0));

        // Swap frontier
        shared_frontier.swap();
        warpFrontier.swap();

        // Perform propagations
        while (!(shared_frontier.isEmpty() && warpFrontier.isEmpty())) {
            // Empty frontier queue if necessary
            const auto shared_queue{shared_frontier.getIsInQueue()};
            for (nodeSize_t queuePackIdx{laneIdx}; queuePackIdx < packedQueueSize; queuePackIdx += KernelUtils::WARP_SIZE) {
                shared_queue[queuePackIdx] = 0;
            }

            // Visit all neighbors of nodes in the current shared frontier (using stride access)
            const auto sharedFrontierSize{shared_frontier.getSize()};
            const auto* __restrict__ const sharedFrontierValues{shared_frontier.getValues()};
            for (nodeSize_t frontierIdx{laneIdx}; frontierIdx < sharedFrontierSize; frontierIdx += KernelUtils::WARP_SIZE) {
                // Get node index
                const auto nodeIdx{sharedFrontierValues[frontierIdx]};

                // Process node
                processNode(nodeIdx, &warpFrontier, &shared_frontier, d_neighborOffsets, d_neighborValues, d_connectedComponentLocalIndexMappings, d_currentCosts);
            }

            // Sync frontier sizes
            shared_frontier.setQueueSize(::__shfl_sync(KernelUtils::BROADCAST_SHUFFLE_MASK, shared_frontier.getQueueSize(), 0));
            warpFrontier.setQueueSize(::__shfl_sync(KernelUtils::BROADCAST_SHUFFLE_MASK, warpFrontier.getQueueSize(), 0));

            // Visit all neighbors of nodes in the current global frontier (using stride access)
            const auto warpFrontierSize{warpFrontier.getSize()};
            const auto* __restrict__ const warpFrontierValues{warpFrontier.getValues()};
            for (nodeSize_t frontierIdx{laneIdx}; frontierIdx < warpFrontierSize; frontierIdx += KernelUtils::WARP_SIZE) {
                // Get node index
                const auto nodeIdx{warpFrontierValues[frontierIdx]};

                // Process node
                processNode(nodeIdx, &warpFrontier, &shared_frontier, d_neighborOffsets, d_neighborValues, d_connectedComponentLocalIndexMappings, d_currentCosts);
            }

            // Sync frontier sizes
            shared_frontier.setQueueSize(::__shfl_sync(KernelUtils::BROADCAST_SHUFFLE_MASK, shared_frontier.getQueueSize(), 0));
            warpFrontier.setQueueSize(::__shfl_sync(KernelUtils::BROADCAST_SHUFFLE_MASK, warpFrontier.getQueueSize(), 0));

            // Swap frontier
            shared_frontier.swap();
            warpFrontier.swap();
        }
    }

    __global__ void SequenceGraphKernels::cooperativeInsertionsAndPropagations(const edgeSize_t* __restrict__ const d_neighborOffsets, const nodeSize_t* __restrict__ const d_neighborValues, cost_t* __restrict__ const d_currentCosts, nodeSize_t* __restrict__ const d_frontierBuffer1, nodeSize_t* __restrict__ const d_frontierBuffer2, int* __restrict__ const d_frontierQueue1, int* __restrict__ const d_frontierQueue2, nodeSize_t* __restrict__ const d_frontierBufferSize1, nodeSize_t* __restrict__ const d_frontierBufferSize2, const bool earlyExit) {
        // Get grid handler
        const auto grid{::cooperative_groups::this_grid()};

        // Get selector
        bool selector{true};

        // Loop while frontier not empty
        while (true) {
            // Get frontier size
            const auto frontierSize{(selector)? *d_frontierBufferSize1 : *d_frontierBufferSize2};

            // Check if frontier is empty
            if (frontierSize == 0) {
                break;
            }

            // Get frontier buffer
            const auto* __restrict__ const frontierBuffer{(selector)? d_frontierBuffer1 : d_frontierBuffer2};

            // Get queue buffer
            auto* __restrict__ const queueBuffer{(selector)? d_frontierBuffer2 : d_frontierBuffer1};

            // Get current and next queue masks
            auto* __restrict__ const currentQueueMask{(selector)? d_frontierQueue1 : d_frontierQueue2};
            auto* __restrict__ const nextQueueMask{(selector)? d_frontierQueue2 : d_frontierQueue1};

            // Get queue size pointer
            auto* __restrict__ const queueSize{(selector)? d_frontierBufferSize2 : d_frontierBufferSize1};

            // Get thread ID
            const auto threadID{::blockIdx.x * ::blockDim.x + ::threadIdx.x};

            // Process all nodes in the frontier (using grid-stride loop)
            const auto stride{::gridDim.x * ::blockDim.x};
            for (nodeSize_t frontierIdx{threadID}; frontierIdx < frontierSize; frontierIdx += stride) {
                // Get node index
                const auto nodeIdx{frontierBuffer[frontierIdx]};

                // Clear node from the queue
                currentQueueMask[nodeIdx] = false;

                // Skip rest of operations if early exit is set to true
                if (earlyExit) {
                    continue;
                }

                // Get updated current layer neighbor cost
                const auto updatedCurrentLayerNeighborCost{d_currentCosts[nodeIdx] + SequenceGraph::INSERTION_COST};

                // Loop over all neighbors and search for propagations
                const auto neighborsStart{d_neighborOffsets[nodeIdx]};
                const auto neighborEnd{d_neighborOffsets[nodeIdx + 1]};
                for (auto neighborOffset{neighborsStart}; neighborOffset < neighborEnd; ++neighborOffset) {
                    // Get neighbor index
                    const auto neighborIdx{d_neighborValues[neighborOffset]};

                    // Set cost to atomic min and get previous current layer neighbor cost and check for improvement
                    if (const auto previousCurrentLayerNeighborCost{::atomicMin(d_currentCosts + neighborIdx, updatedCurrentLayerNeighborCost)}; (updatedCurrentLayerNeighborCost < previousCurrentLayerNeighborCost) && (!::atomicExch(nextQueueMask + neighborIdx, 1))) {
                        // Get active threads
                        const auto active{::cooperative_groups::coalesced_threads()};

                        // Get insertion base
                        nodeSize_t insertionBase{0};

                        // Leader thread reserves space for all threads in the warp
                        if (active.thread_rank() == 0) {
                            insertionBase = ::atomicAdd(queueSize, active.size());
                        }

                        // Shuffle insertion base
                        insertionBase = active.shfl(insertionBase, 0);

                        // Insert in queue
                        queueBuffer[insertionBase + active.thread_rank()] = neighborIdx;
                    }
                }
            }

            // Synchronize grid
            grid.sync();

            // Flush current size
            if (::cooperative_groups::grid_group::thread_rank() == 0) {
                if (selector) {
                    *d_frontierBufferSize1 = 0;
                }
                else {
                    *d_frontierBufferSize2 = 0;
                }
            }

            // Swap selector
            selector = !selector;
        }
    }

    __global__ void SequenceGraphKernels::cooperativeBlockAggregationInsertionsAndPropagations(const edgeSize_t* __restrict__ const d_neighborOffsets, const nodeSize_t* __restrict__ const d_neighborValues, cost_t* __restrict__ const d_currentCosts, nodeSize_t* __restrict__ const d_frontierBuffer1, nodeSize_t* __restrict__ const d_frontierBuffer2, int* __restrict__ const d_frontierQueue1, int* __restrict__ const d_frontierQueue2, nodeSize_t* __restrict__ const d_frontierBufferSize1, nodeSize_t* __restrict__ const d_frontierBufferSize2, const bool earlyExit) {
        // Shared memory queue
        __shared__ nodeSize_t shared_queueSize;
        __shared__ nodeSize_t shared_queueBuffer[SHARED_QUEUE_BUFFER_SIZE];
        __shared__ nodeSize_t shared_globalInsertionBase;

        // Get grid handler
        const auto grid{::cooperative_groups::this_grid()};

        // Get selector
        bool selector{true};

        // Loop while frontier not empty
        while (true) {
            // Get frontier size
            const auto frontierSize{(selector)? *d_frontierBufferSize1 : *d_frontierBufferSize2};

            // Check if frontier is empty
            if (frontierSize == 0) {
                break;
            }

            // Get frontier buffer
            const auto* __restrict__ const frontierBuffer{(selector)? d_frontierBuffer1 : d_frontierBuffer2};

            // Get queue buffer
            auto* __restrict__ const queueBuffer{(selector)? d_frontierBuffer2 : d_frontierBuffer1};

            // Get current and next queue masks
            auto* __restrict__ const currentQueueMask{(selector)? d_frontierQueue1 : d_frontierQueue2};
            auto* __restrict__ const nextQueueMask{(selector)? d_frontierQueue2 : d_frontierQueue1};

            // Get queue size pointer
            auto* __restrict__ const queueSize{(selector)? d_frontierBufferSize2 : d_frontierBufferSize1};

            // Initialize block queue size
            if (::threadIdx.x == 0) {
                shared_queueSize = 0;
            }

            // Synchronize block
            cooperative_groups::thread_block::sync();

            // Get thread ID
            const auto threadID{::blockIdx.x * ::blockDim.x + ::threadIdx.x};

            // Process all nodes in the frontier (using grid-stride loop)
            const auto stride{::gridDim.x * ::blockDim.x};
            for (nodeSize_t frontierIdx{threadID}; frontierIdx < frontierSize; frontierIdx += stride) {
                // Get node index
                const auto nodeIdx{frontierBuffer[frontierIdx]};

                // Clear node from the queue
                currentQueueMask[nodeIdx] = false;

                // Skip rest of operations if early exit is set to true
                if (earlyExit) {
                    continue;
                }

                // Get updated current layer neighbor cost
                const auto updatedCurrentLayerNeighborCost{d_currentCosts[nodeIdx] + SequenceGraph::INSERTION_COST};

                // Loop over all neighbors and search for propagations
                const auto neighborsStart{d_neighborOffsets[nodeIdx]};
                const auto neighborEnd{d_neighborOffsets[nodeIdx + 1]};
                for (auto neighborOffset{neighborsStart}; neighborOffset < neighborEnd; ++neighborOffset) {
                    // Get neighbor index
                    const auto neighborIdx{d_neighborValues[neighborOffset]};

                    // Set cost to atomic min and get previous current layer neighbor cost and check for improvement
                    if (const auto previousCurrentLayerNeighborCost{::atomicMin(d_currentCosts + neighborIdx, updatedCurrentLayerNeighborCost)}; (updatedCurrentLayerNeighborCost < previousCurrentLayerNeighborCost) && (!::atomicExch(nextQueueMask + neighborIdx, 1))) {
                        // Get active threads
                        const auto active{::cooperative_groups::coalesced_threads()};

                        // Get insertion base
                        nodeSize_t insertionBase{0};

                        // Leader thread reserves space for all threads in the warp
                        if (active.thread_rank() == 0) {
                            insertionBase = ::atomicAdd(&shared_queueSize, active.size());
                        }

                        // Shuffle insertion base
                        insertionBase = active.shfl(insertionBase, 0);

                        // Check for shared queue overflow and fall back to inserting directly in global queue if necessary
                        if (insertionBase + active.size() > SHARED_QUEUE_BUFFER_SIZE) {
                            // Leader thread reserves space for all threads in the warp
                            if (active.thread_rank() == 0) {
                                insertionBase = ::atomicAdd(queueSize, active.size());
                            }

                            // Shuffle insertion base
                            insertionBase = active.shfl(insertionBase, 0);

                            // Insert in queue
                            queueBuffer[insertionBase + active.thread_rank()] = neighborIdx;
                        } else {
                            // Insert in shared queue
                            shared_queueBuffer[insertionBase + active.thread_rank()] = neighborIdx;
                        }
                    }
                }
            }

            // Synchronize block
            cooperative_groups::thread_block::sync();

            // Flush shared memory queue to global queue
            if ((::threadIdx.x == 0) && (shared_queueSize > 0)) {
                shared_globalInsertionBase = ::atomicAdd(queueSize, shared_queueSize);
            }

            // Synchronize block
            cooperative_groups::thread_block::sync();

            // Copy items from shared memory block queue to global queue buffer
            for (targetSize_t threadIdx{::threadIdx.x}; threadIdx < shared_queueSize; threadIdx += ::blockDim.x) {
                queueBuffer[shared_globalInsertionBase + threadIdx] = shared_queueBuffer[threadIdx]; // NOLINT
            }

            // Reset queue size
            if (::threadIdx.x == 0) {
                shared_queueSize = 0;
            }

            // Synchronize grid
            grid.sync();

            // Flush current size
            if (::cooperative_groups::grid_group::thread_rank() == 0) {
                if (selector) {
                    *d_frontierBufferSize1 = 0;
                }
                else {
                    *d_frontierBufferSize2 = 0;
                }
            }

            // Swap selector
            selector = !selector;
        }
    }

    __global__ void SequenceGraphKernels::minCost(const cost_t* __restrict__ const d_currentCosts, cost_t* __restrict__ const d_scores, const nodeSize_t numNodes, const scoreSize_t scoreIdx) {
        // Shared memory array to store the warp-level minima inside the block
        extern __shared__ cost_t shared_minCosts[];

        // Get thread id
        const auto threadId{::blockIdx.x * ::blockDim.x + ::threadIdx.x};

        // Initialize thread-level minimum
        auto threadMinCost{SequenceGraph::COST_MAX_VALUE};

        // Grid-level reduction using stride (safe for overflowing threads)
        const auto stride{::blockDim.x * ::gridDim.x};
        for (nodeSize_t nodeIdx{threadId}; nodeIdx < numNodes; nodeIdx += stride) {
            const cost_t cost{d_currentCosts[nodeIdx]};
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
        if (warpIdx != 0) {
            return;
        }

        // Get warp-level minimum
        const auto numWarpsPerBlock{(::blockDim.x + KernelUtils::WARP_SIZE - 1) >> KernelUtils::WARP_SHIFT};
        threadMinCost = (::threadIdx.x < numWarpsPerBlock)? shared_minCosts[::threadIdx.x] : SequenceGraph::COST_MAX_VALUE;

        // Block-level reduction using warp shuffling
        const auto activeMask{::__activemask()};
#pragma unroll
        for (auto offset{KernelUtils::WARP_SIZE / 2}; offset > 0; offset >>= 1) {
            const auto shuffledValue{::__shfl_down_sync(activeMask, threadMinCost, offset)};
            threadMinCost = ::min(shuffledValue, threadMinCost);
        }

        // Final grid-level reduction using the first thread of each block
        if (laneIdx != 0) {
            return;
        }
        ::atomicMin(d_scores + scoreIdx, threadMinCost);
    }
} // cuSGA