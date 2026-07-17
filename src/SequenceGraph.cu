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
            costsDoubleBuffer->d_swap();

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
                    costsDoubleBuffer->d_swap();
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

    __host__ void SequenceGraph::initialize(const bool sync) const {
        // Launch initialize kernel
        KernelUtils::cudaLaunchKernel<SequenceGraphKernels::initialize>(SequenceGraphKernels::INITIALIZE_DYNAMIC_SMEM_SIZE, SequenceGraphKernels::INITIALIZE_DYNAMIC_SMEM_SIZE, pangenomeGraph.getNumNodes(), sync, *pinned_instance);
    }

    __host__ void SequenceGraph::substitutions(const ::size_t layerIdx, const bool sync) const {
        // Launch substitutions kernel
        KernelUtils::cudaLaunchKernel<SequenceGraphKernels::substitutions>(SequenceGraphKernels::SUBSTITUTIONS_DYNAMIC_SMEM_SIZE, SequenceGraphKernels::SUBSTITUTIONS_MAX_BLOCK_SIZE, pangenomeGraph.getNumNodes(), sync, *pinned_instance, layerIdx);
    }

    __host__ void SequenceGraph::deletions(const ::size_t layerIdx, const bool sync) const {
        // Launch deletions kernel
        KernelUtils::cudaLaunchKernel<SequenceGraphKernels::deletions>(SequenceGraphKernels::DELETIONS_DYNAMIC_SMEM_SIZE, SequenceGraphKernels::DELETIONS_MAX_BLOCK_SIZE, pangenomeGraph.getNumNodes(), sync, *pinned_instance, layerIdx);
    }

    __host__ void SequenceGraph::insertions(const Frontier& d_frontier, const bool sync) const {
        // Launch insertions kernel
        KernelUtils::cudaLaunchKernel<SequenceGraphKernels::insertions>(SequenceGraphKernels::INSERTIONS_DYNAMIC_SMEM_SIZE, SequenceGraphKernels::INSERTIONS_MAX_BLOCK_SIZE, pangenomeGraph.getNumNodes(), sync, *pinned_instance, d_frontier);
    }

    __host__ void SequenceGraph::propagations(const Frontier& d_frontier, const ::size_t layerIdx, const bool sync) const {
        // Launch propagations kernel
        KernelUtils::cudaLaunchKernel<SequenceGraphKernels::propagations>(SequenceGraphKernels::PROPAGATIONS_DYNAMIC_SMEM_SIZE, SequenceGraphKernels::PROPAGATIONS_MAX_BLOCK_SIZE, d_frontier.getSize(), sync, *pinned_instance, d_frontier, layerIdx);
    }

    __host__ void SequenceGraph::minScore(const bool sync) const {
        // Launch min score kernel
        KernelUtils::cudaLaunchKernel<SequenceGraphKernels::minScore>(SequenceGraphKernels::MIN_SCORE_DYNAMIC_SMEM_SIZE, SequenceGraphKernels::MIN_SCORE_MAX_BLOCK_SIZE, pangenomeGraph.getNumNodes(), sync, *pinned_instance);
    }

    __global__ void SequenceGraphKernels::initialize(const SequenceGraph d_sequenceGraph) { // NOLINT
        // Get thread node index
        const auto nodeIdx{::blockIdx.x * ::blockDim.x + ::threadIdx.x};

        // Get pangenome graph
        const auto pangenomeGraph{d_sequenceGraph.getPangenomeGraph()};

        // Check for thread overflow
        if (const auto numNodes{pangenomeGraph->getNumNodes()}; nodeIdx < numNodes) {
            // Get costs double buffer
            const auto costsDoubleBuffer{d_sequenceGraph.getCostsDoubleBuffer()};

            // Get current DNA base in the sequence
            const auto sequence{d_sequenceGraph.getSequence()};
            const auto currentBase{sequence[0]};

            // Check if match
            const auto isMatch{pangenomeGraph->getDNABase(nodeIdx) == currentBase};

            // Compute updated node cost
            const auto updatedCost{(isMatch)? 0 : SequenceGraph::INITIALIZATION_COST};

            // Initialize node cost for the next layer
            costsDoubleBuffer->current()[nodeIdx].store(updatedCost, ::cuda::memory_order_relaxed);
        }
    }

    __global__ void SequenceGraphKernels::substitutions(const SequenceGraph d_sequenceGraph, const ::size_t layerIdx) { // NOLINT
        // Get thread node index
        const auto nodeIdx{::blockIdx.x * ::blockDim.x + ::threadIdx.x};

        // Get pangenome graph
        const auto pangenomeGraph{d_sequenceGraph.getPangenomeGraph()};

        // Check for thread overflow
        if (const auto numNodes{pangenomeGraph->getNumNodes()}; nodeIdx < numNodes) {
            // Get costs double buffer
            const auto costsDoubleBuffer{d_sequenceGraph.getCostsDoubleBuffer()};

            // Get neighbors
            const auto neighbors{pangenomeGraph->getNeighbors(nodeIdx)};
            const auto numNeighbors{pangenomeGraph->getNumNeighbors(nodeIdx)};

            // Get node cost in the previous layer
            const auto previousLayerCost{costsDoubleBuffer->alternate()[nodeIdx].load(::cuda::memory_order_relaxed)};

            // Get current DNA base in the sequence
            const auto sequence{d_sequenceGraph.getSequence()};
            const auto currentBase{(*sequence)[layerIdx]};

            // Loop over all neighbors
            for (::size_t neighborOffset{0}; neighborOffset < numNeighbors; ++neighborOffset) {
                // Get neighbor index
                const auto neighborIdx{neighbors[neighborOffset]};

                // Get node cost in the current layer
                const auto currentLayerCost{costsDoubleBuffer->current()[neighborIdx].load(::cuda::memory_order_relaxed)};

                // Compute updated node cost and check for improvement
                const auto isMatch{pangenomeGraph->getDNABase(neighborIdx) == currentBase};
                if (const auto updatedCost{(isMatch)? previousLayerCost : previousLayerCost + SequenceGraph::SUBSTITUTION_COST}; updatedCost < currentLayerCost) {
                    // Set cost to atomic min
                    costsDoubleBuffer->current()[neighborIdx].fetch_min(updatedCost);
                }
            }
        }
    }

    __global__ void SequenceGraphKernels::deletions(const SequenceGraph d_sequenceGraph) { // NOLINT
        // Get thread node index
        const auto nodeIdx{::blockIdx.x * ::blockDim.x + ::threadIdx.x};

        // Get pangenome graph
        const auto pangenomeGraph{d_sequenceGraph.getPangenomeGraph()};

        // Check for thread overflow
        if (const auto numNodes{pangenomeGraph->getNumNodes()}; nodeIdx < numNodes) {
            // Get costs double buffer
            const auto costsDoubleBuffer{d_sequenceGraph.getCostsDoubleBuffer()};

            // Get node cost in the previous layer
            const auto previousLayerCost{costsDoubleBuffer->alternate()[nodeIdx].load(::cuda::memory_order_relaxed)};

            // Compute updated node cost
            const auto updatedCost{previousLayerCost + SequenceGraph::DELETION_COST};

            // Initialize node cost for the next layer
            costsDoubleBuffer->current()[nodeIdx].store(updatedCost, ::cuda::memory_order_relaxed);
        }
    }

    __global__ void SequenceGraphKernels::insertions(const SequenceGraph d_sequenceGraph, const Frontier d_frontier) { // NOLINT
        // Get thread node index
        const auto nodeIdx{::blockIdx.x * ::blockDim.x + ::threadIdx.x};

        // Check for thread overflow
        if (const auto numNodes{d_sequenceGraph.getPangenomeGraph()->getNumNodes()}; nodeIdx < numNodes) {
            // Insert node in the frontier without queueing
            d_frontier.insertWithoutQueueing(nodeIdx);

            // Set initial frontier size
            if (nodeIdx == 0) {
                d_frontier.setSize(numNodes);
            }
        }
    }

    __global__ void SequenceGraphKernels::propagations(const SequenceGraph d_sequenceGraph, const Frontier d_frontier, const ::size_t layerIdx) {
        // Get thread node index
        const auto threadId{::blockIdx.x * ::blockDim.x + ::threadIdx.x};

        // Get pangenome graph
        const auto pangenomeGraph{d_sequenceGraph.getPangenomeGraph()};

        // Get current DNA base in the sequence
        const auto sequence{d_sequenceGraph.getSequence()};
        const auto currentBase{(*sequence)[layerIdx]};

        // Check for thread overflow and if given node is a root in the corresponding character graph for the given layer
        if (const auto frontierSize{d_frontier->getSize()}; threadId < frontierSize) {
            // Get node index
            const auto nodeIdx{d_frontier->getValue(threadId)};

            // Get costs double buffer
            const auto costsDoubleBuffer{d_sequenceGraph->getCostsDoubleBuffer()};

            // Get neighbors
            const auto neighbors{pangenomeGraph->getNeighbors(nodeIdx)};
            const auto numNeighbors{pangenomeGraph->getNumNeighbors(nodeIdx)};

            // Get node cost in the current layer
            const auto currentLayerCost{costsDoubleBuffer->current()[nodeIdx].load(::cuda::memory_order_relaxed)};

            // Loop over all neighbors
            for (::size_t neighborOffset{0}; neighborOffset < numNeighbors; ++neighborOffset) {
                // Get neighbor index
                const auto neighborIdx{neighbors[neighborOffset]};

                // Check if neighbor base value is different from current DNA base in the sequence (edge is in the character graph)
                if (const auto neighborBaseValue{pangenomeGraph->getDNABase(neighborIdx)}; neighborBaseValue != currentBase) {
                    // Get node cost in the current layer
                    const auto currentLayerNeighborCost{costsDoubleBuffer->current()[neighborIdx].load(::cuda::memory_order_relaxed)};

                    // Compute updated node cost and check for improvement
                    if (const auto updatedCost{currentLayerCost + SequenceGraph::INSERTION_COST}; updatedCost < currentLayerNeighborCost) {
                        // Set cost to atomic min
                        if (const auto previousNeighborCost{costsDoubleBuffer->current()[neighborIdx].fetch_min(updatedCost)}; updatedCost < previousNeighborCost) {
                            d_frontier->atomicInsertAndGrow(neighborIdx);
                        }
                    }
                }
            }
        }
    }

    __global__ void SequenceGraphKernels::minScore(SequenceGraph* const sequenceGraph) {
        // Get thread id
        const auto threadId{::blockIdx.x * ::blockDim.x + ::threadIdx.x};

        // Get pangenome graph
        const auto pangenomeGraph{sequenceGraph->getPangenomeGraph()};

        // Get costs
        const auto costs{sequenceGraph->getCostsDoubleBuffer()->current()};

        // Shared memory array to store the warp-level minima inside the block
        __shared__ ::uint64_t warpMins[KernelUtils::MAX_WARPS_PER_BLOCK];

        // Initialize thread-level minimum
        auto threadMin{SequenceGraph::SCORE_MAX_VALUE};

        // Grid-level reduction using stride (safe for overflowing threads)
        const auto stride{::blockDim.x * ::gridDim.x};
        const auto numNodes{pangenomeGraph->getNumNodes()};
        for (::size_t i{threadId}; i < numNodes; i += stride) {
            if (const ::uint64_t cost{costs[i].load(::cuda::memory_order_relaxed)}; cost < threadMin) {
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
            const auto numWarpsPerBlock{(::blockDim.x + KernelUtils::WARP_SIZE - 1) / KernelUtils::WARP_SIZE};
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
                sequenceGraph->getAtomicScore().fetch_min(threadMin, ::cuda::memory_order_relaxed);
            }
        }
    }
} // cuSGA