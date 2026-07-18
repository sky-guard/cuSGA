#include "SequenceGraph.cuh"

#include <fstream>

#include "Frontier.cuh"
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
            minScore(true);

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

    __global__ void SequenceGraphKernels::initialize(const SequenceGraph d_sequenceGraph) { // NOLINT
        // Get thread node index and check for thread overflow
        if (const auto nodeIdx{::blockIdx.x * ::blockDim.x + ::threadIdx.x}; nodeIdx < d_sequenceGraph.getPangenomeGraph().getNumNodes()) {
            // Get current DNA base in the sequence
            const auto sequenceBase{d_sequenceGraph.getSequence()[0]};

            // Get DNA base for the current node
            const auto nodeBase{d_sequenceGraph.getPangenomeGraph().getDNABase(nodeIdx)};

            // Compute updated node cost
            const auto initialNodeCost{(sequenceBase == nodeBase)? 0 : SequenceGraph::INITIALIZATION_COST};

            // Initialize node cost for the next layer
            d_sequenceGraph.getCostsDoubleBuffer().current()[nodeIdx].store(initialNodeCost, ::cuda::memory_order_relaxed);
        }
    }

    __global__ void SequenceGraphKernels::substitutions(const SequenceGraph d_sequenceGraph, const ::size_t layerIdx) { // NOLINT
        // Get thread node index and check for thread overflow
        if (const auto nodeIdx{::blockIdx.x * ::blockDim.x + ::threadIdx.x}; nodeIdx < d_sequenceGraph.getPangenomeGraph().getNumNodes()) {
            // Get current DNA base in the sequence
            const auto sequenceBase{d_sequenceGraph.getSequence()[layerIdx]};

            // Get node cost in the previous layer
            const auto previousLayerNodeCost{d_sequenceGraph.getCostsDoubleBuffer().alternate()[nodeIdx].load(::cuda::memory_order_relaxed)};

            // Loop over all neighbors
            const auto neighbors{d_sequenceGraph.getPangenomeGraph().getNeighbors(nodeIdx)};
            const auto numNeighbors{d_sequenceGraph.getPangenomeGraph().getNumNeighbors(nodeIdx)};
            for (::size_t neighborOffset{0}; neighborOffset < numNeighbors; ++neighborOffset) {
                // Get neighbor index
                const auto neighborIdx{neighbors[neighborOffset]};

                // Get DNA base for the neighbor
                const auto neighborBase{d_sequenceGraph.getPangenomeGraph().getDNABase(neighborIdx)};

                // Get node cost in the current layer
                const auto currentLayerNeighborCost{d_sequenceGraph.getCostsDoubleBuffer().current()[neighborIdx].load(::cuda::memory_order_relaxed)};

                // Compute updated node cost and check for improvement
                if (const auto updatedCurrentLayerNeighborCost{(sequenceBase == neighborBase)? previousLayerNodeCost : previousLayerNodeCost + SequenceGraph::SUBSTITUTION_COST}; updatedCurrentLayerNeighborCost < currentLayerNeighborCost) {
                    // Set cost to atomic min
                    d_sequenceGraph.getCostsDoubleBuffer().current()[neighborIdx].fetch_min(updatedCurrentLayerNeighborCost);
                }
            }
        }
    }

    __global__ void SequenceGraphKernels::deletions(const SequenceGraph d_sequenceGraph) { // NOLINT
        // Get thread node index and check for thread overflow
        if (const auto nodeIdx{::blockIdx.x * ::blockDim.x + ::threadIdx.x}; nodeIdx < d_sequenceGraph.getPangenomeGraph().getNumNodes()) {
            // Get node cost in the previous layer
            const auto previousLayerNodeCost{d_sequenceGraph.getCostsDoubleBuffer().alternate()[nodeIdx].load(::cuda::memory_order_relaxed)};

            // Compute updated node cost
            const auto currentLayerNodeCost{previousLayerNodeCost + SequenceGraph::DELETION_COST};

            // Initialize node cost for the next layer
            d_sequenceGraph.getCostsDoubleBuffer().current()[nodeIdx].store(currentLayerNodeCost, ::cuda::memory_order_relaxed);
        }
    }

    __global__ void SequenceGraphKernels::insertions(const SequenceGraph d_sequenceGraph, const Frontier d_frontier) { // NOLINT
        // Get thread node index and check for thread overflow
        if (const auto nodeIdx{::blockIdx.x * ::blockDim.x + ::threadIdx.x}; nodeIdx < d_sequenceGraph.getPangenomeGraph().getNumNodes()) {
            // Set initial frontier size
            if (nodeIdx == 0) {
                d_frontier.d2d_setSize(d_sequenceGraph.getPangenomeGraph().getNumNodes());
            }

            // Insert node in the frontier without queueing
            d_frontier.insertWithoutQueueing(nodeIdx);
        }
    }

    __global__ void SequenceGraphKernels::propagations(const SequenceGraph d_sequenceGraph, const Frontier d_frontier, const ::size_t layerIdx) { // NOLINT
        // Get thread ID and check for thread overflow
        if (const auto threadId{::blockIdx.x * ::blockDim.x + ::threadIdx.x}; threadId < d_frontier.getSize()) {
            // Get node index
            const auto nodeIdx{d_frontier.getValue(threadId)};

            // Get node cost in the current layer
            const auto currentLayerNodeCost{d_sequenceGraph.getCostsDoubleBuffer().current()[nodeIdx].load(::cuda::memory_order_relaxed)};

            // Loop over all neighbors
            const auto neighbors{d_sequenceGraph.getPangenomeGraph().getNeighbors(nodeIdx)};
            const auto numNeighbors{d_sequenceGraph.getPangenomeGraph().getNumNeighbors(nodeIdx)};
            for (::size_t neighborOffset{0}; neighborOffset < numNeighbors; ++neighborOffset) {
                // Get neighbor index
                const auto neighborIdx{neighbors[neighborOffset]};

                // Get node cost in the current layer
                const auto currentLayerNeighborCost{d_sequenceGraph.getCostsDoubleBuffer().current()[neighborIdx].load(::cuda::memory_order_relaxed)};

                // Compute updated node cost and check for improvement
                if (const auto updatedCurrentLayerNeighborCost{currentLayerNodeCost + SequenceGraph::INSERTION_COST}; updatedCurrentLayerNeighborCost < currentLayerNeighborCost) {
                    // Set cost to atomic min
                    if (const auto previousNeighborCost{d_sequenceGraph.getCostsDoubleBuffer().current()[neighborIdx].fetch_min(updatedCurrentLayerNeighborCost)}; updatedCurrentLayerNeighborCost < previousNeighborCost) {
                        d_frontier.d2d_atomicInsertAndGrow(neighborIdx);
                    }
                }
            }
        }
    }

    __global__ void SequenceGraphKernels::minScore(const SequenceGraph d_sequenceGraph, const ::size_t scoreIdx) { // NOLINT
        // Shared memory array to store the warp-level minima inside the block
        extern __shared__ ::uint64_t warpMins[];

        // Get thread id
        const auto threadId{::blockIdx.x * ::blockDim.x + ::threadIdx.x};

        // Initialize thread-level minimum
        auto threadMin{SequenceGraph::SCORE_MAX_VALUE};

        // Grid-level reduction using stride (safe for overflowing threads)
        const auto stride{::blockDim.x * ::gridDim.x};
        const auto numNodes{d_sequenceGraph.getPangenomeGraph().getNumNodes()};
        for (::size_t i{threadId}; i < numNodes; i += stride) {
            if (const ::uint64_t cost{d_sequenceGraph.getCostsDoubleBuffer()[i].load(::cuda::memory_order_relaxed)}; cost < threadMin) {
                threadMin = cost;
            }
        }

        // Warp-level reduction using warp shuffling (safe for overflowing threads)
        const auto warpIdx{::threadIdx.x / KernelUtils::WARP_SIZE};
        const auto laneIdx{::threadIdx.x % KernelUtils::WARP_SIZE};
#pragma unroll
        for (auto offset{KernelUtils::WARP_SIZE / 2}; offset > 0; offset /= 2) {
            if (const auto shuffledValue{::__shfl_down_sync(MIN_SCORE_SHUFFLE_MASK, threadMin, offset)}; shuffledValue < threadMin) {
                threadMin = shuffledValue;
            }
        }

        // Store warp-level minimum using the first thread of each warp
        if (laneIdx == 0) {
            warpMins[warpIdx] = threadMin;
        }
        ::__syncthreads();

        // Block-level reduction using the first warp of each block
        if (warpIdx == 0) {
            // Get warp-level minimum
            const auto numWarpsPerBlock{(::blockDim.x + KernelUtils::WARP_SIZE - 1) >> KernelUtils::WARP_SHIFT};
            threadMin = (::threadIdx.x < numWarpsPerBlock)? warpMins[::threadIdx.x] : SequenceGraph::SCORE_MAX_VALUE;

            // Block-level reduction using warp shuffling
#pragma unroll
            for (auto offset{KernelUtils::WARP_SIZE / 2}; offset > 0; offset /= 2) {
                if (const auto shuffledValue{::__shfl_down_sync(MIN_SCORE_SHUFFLE_MASK, threadMin, offset)}; shuffledValue < threadMin) {
                    threadMin = shuffledValue;
                }
            }

            // Final grid-level reduction using the first thread of each block
            if (laneIdx == 0) {
                d_sequenceGraph.getAtomicScore(scoreIdx).fetch_min(threadMin, ::cuda::memory_order_relaxed);
            }
        }
    }
} // cuSGA