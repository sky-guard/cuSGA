#include "SequenceGraph.cuh"

#include <fstream>

#include "Frontier.cuh"
#include "KernelUtils.cuh"


// TODO: Remove after debugging is done.
__global__ void printCosts(const cuSGA::SequenceGraph* const sequenceGraph, const ::size_t layerIdx) {
    const auto nodeIdx{::blockIdx.x * ::blockDim.x + ::threadIdx.x};
    const auto numNodes{sequenceGraph->getPangenomeGraph()->getNumNodes()};
    const auto costs{sequenceGraph->getCostsDoubleBuffer()->current()};

    if (nodeIdx < numNodes) {
        // Cast 64-bit integers to unsigned long long and use %llu
        ::printf("[Layer %llu] Node %u: %llu.\n",
                 static_cast<unsigned long long>(layerIdx),
                 static_cast<unsigned int>(nodeIdx),
                 static_cast<unsigned long long>(costs[nodeIdx].load(::cuda::memory_order_relaxed)));
    }
}

namespace cuSGA {
    __host__ SequenceGraph* SequenceGraph::createFromFiles(const ::std::string& sequenceFileName, const ::std::string& pangenomeGraphFileName, const ::std::string (& characterGraphFileNames)[NUM_BASES], const bool computeCharacterGraphs) {
        // Read sequence from file
        const auto sequence{PackedDNASequence::createFromFile(sequenceFileName)};

        // Read pangenome graph from file
        const auto pangenomeGraph{PangenomeGraph::createFromFile(pangenomeGraphFileName)};

        // Read or compute and store character graphs from files
        PangenomeGraph* characterGraphs[NUM_BASES]{nullptr};
        // TODO: Add back if character graphs are needed.
        // for (::size_t i{0}; i < NUM_BASES; ++i) {
        //     // Get character graph file name
        //     const auto characterGraphFileName{characterGraphFileNames[i]};

        //     if (computeCharacterGraphs) {
        //         // Compute character graph and store to file
        //         ::std::exit(-1);
        //     }
        //     else {
        //         // Read character graph from file
        //         ::std::ifstream characterGraphFile{characterGraphFileName};
        //         characterGraphs[i] = PangenomeGraph::createFromFile(characterGraphFileName, characterGraphFile, pangenomeGraph->getNumNodes(), ::std::nullopt, pangenomeGraph->getBaseValues());
        //     }
        // }

        // Create costs double buffer instance
        const auto numNodes{pangenomeGraph->getNumNodes()};
        const auto costsDoubleBuffer{DoubleBuffer<::cuda::atomic<::uint64_t, ::cuda::thread_scope_device>>::create(numNodes)};

        // Create sequence graph instance
        const auto sequenceGraph{new SequenceGraph{sequence, pangenomeGraph, characterGraphs, costsDoubleBuffer}};

        return sequenceGraph;
    }

    __host__ SequenceGraph* SequenceGraph::copyToDevice() {
        // Check if device instance already exists for this sequence graph
        if (d_instance) {
            return d_instance;
        }

        // Copy buffers data from host to device
        const auto d_sequence{sequence->copyToDevice()};
        const auto d_pangenomeGraph{pangenomeGraph->copyToDevice()};
        PangenomeGraph* d_characterGraphs[NUM_BASES]{nullptr};
        // TODO: Add back if character graphs are needed.
        // for (::size_t i{0}; i < NUM_BASES; ++i) {
        //     d_characterGraphs[i] = characterGraphs[i]->copyToDevice();
        // }
        const auto d_costsDoubleBuffer{costsDoubleBuffer->copyToDevice()};

        // Allocate device pangenome graph instance
        SequenceGraph* d_sequenceGraph{nullptr};
        KernelUtils::cudaMalloc(&d_sequenceGraph, sizeof(SequenceGraph));

        // Create temporary host instance holding the device pointers
        const SequenceGraph deviceSequenceGraph{d_sequence, d_pangenomeGraph, d_characterGraphs, d_costsDoubleBuffer, score.load(::cuda::memory_order_relaxed), d_instance};

        // Update host instance data
        this->d_instance = d_sequenceGraph;

        // Copy instance data from host to device
        KernelUtils::cudaMemcpy(d_sequenceGraph, &deviceSequenceGraph, sizeof(SequenceGraph), ::cudaMemcpyHostToDevice);

        return d_sequenceGraph;
    }

    __host__ void SequenceGraph::free() const {
        // Free device memory if device instance is present
        if (d_instance) {
            // Free device sequence graph instance
            KernelUtils::cudaFree(d_instance);
        }

        // Free host memory
        if (sequence) {
            this->sequence->free();
        }
        for (::size_t i{0}; i < NUM_BASES; ++i) {
            if (characterGraphs[i]) {
                this->characterGraphs[i]->free();
            }
        }
        if (pangenomeGraph) {
            this->pangenomeGraph->free(true);
        }
        if (costsDoubleBuffer) {
            this->costsDoubleBuffer->free();
        }
        delete this;
    }

    __host__ __device__ PackedDNASequence* SequenceGraph::getSequence() const {
        return sequence;
    }

    __host__ __device__ PangenomeGraph* SequenceGraph::getPangenomeGraph() const {
        return pangenomeGraph;
    }

    __host__ __device__ PangenomeGraph* SequenceGraph::getCharacterGraph(const DNABase base) const {
        return characterGraphs[static_cast<::uint8_t>(base)];
    }

    __host__ __device__ DoubleBuffer<::cuda::atomic<::uint64_t, ::cuda::thread_scope_device>>* SequenceGraph::getCostsDoubleBuffer() const {
        return costsDoubleBuffer;
    }

    __host__ __device__ ::uint64_t SequenceGraph::getScore() const {
        return score.load(::cuda::memory_order_relaxed);
    }

    __host__ ::uint64_t SequenceGraph::getScoreSync() {
        // Copy back score from device
        if (d_instance) {
            KernelUtils::cudaMemcpy(&this->score, &d_instance->score, sizeof(score),::cudaMemcpyDeviceToHost);
        }

        return score.load(::cuda::memory_order_relaxed);
    }

    __host__ __device__ ::cuda::atomic<::uint64_t, ::cuda::thread_scope_device>& SequenceGraph::getAtomicScore() {
        return score;
    }

    __host__ __device__ SequenceGraph* SequenceGraph::getDeviceInstance() const {
        return d_instance;
    }

    __host__ uint64_t SequenceGraph::align() {
        // Check for non-empty sequence
        if (sequence->getNumBases() == 0) {
            throw ::std::runtime_error{"Unable to align an empty sequence!"};
        }

        // Copy sequence graph instance to device
        copyToDevice();

        // Create frontier instance
        const auto frontier{Frontier::create(pangenomeGraph->getNumNodes())};

        // Copy frontier to device
        frontier->copyToDevice();

        // Perform initialization step
        initialize(false);

        // Swap costs double buffer for the next layer
        costsDoubleBuffer->swapSync();

        // Solve alignment layer by layer
        for (::size_t layerIdx{1}; layerIdx < sequence->getNumBases(); ++layerIdx) {
            // Perform deletions for the given layer
            deletions(layerIdx,false);

            // Perform substitutions for the given layer
            substitutions(layerIdx, false);

            // Skip insertions and propagations for the last layer
            if (layerIdx < sequence->getNumBases() - 1) {
                // Empty frontier
                frontier->emptySync();

                // Perform insertions for the given layer
                insertions(frontier, false);

                // Perform propagations for the given layer
                while (!frontier->isEmptySync()) {
                    propagations(layerIdx, frontier, false);
                    frontier->swapToQueueSync();
                }

                // TODO: Remove after debugging is done.
                // KernelUtils::launchKernel<printCosts>(0, 0, pangenomeGraph->getNumNodes(), true, d_instance, layerIdx);

                // Swap costs double buffer for the next layer
                costsDoubleBuffer->swapSync();
            }
            // TODO: Remove after debugging is done.
            else {
                // KernelUtils::launchKernel<printCosts>(0, 0, pangenomeGraph->getNumNodes(), true, d_instance, layerIdx);
            }
        }

        // Delete frontier instance
        frontier->free();

        // Compute minimum score
        minScore(true);

        // Copy back score from device
        const auto score{getScoreSync()};

        return score;
    }

    __host__ void SequenceGraph::initialize(const bool sync) {
        // Launch initialize kernel
        KernelUtils::cudaLaunchKernel<SequenceGraphKernels::initialize>(SequenceGraphKernels::INITIALIZE_DYNAMIC_SMEM_SIZE, SequenceGraphKernels::INITIALIZE_DYNAMIC_SMEM_SIZE, pangenomeGraph->getNumNodes(), sync, d_instance);
    }

    __host__ void SequenceGraph::substitutions(const ::size_t layerIdx, const bool sync) const {
        // Launch substitutions kernel
        KernelUtils::cudaLaunchKernel<SequenceGraphKernels::substitutions>(SequenceGraphKernels::SUBSTITUTIONS_DYNAMIC_SMEM_SIZE, SequenceGraphKernels::SUBSTITUTIONS_MAX_BLOCK_SIZE, pangenomeGraph->getNumNodes(), sync, d_instance, layerIdx);
    }

    __host__ void SequenceGraph::deletions(const ::size_t layerIdx, const bool sync) const {
        // Launch deletions kernel
        KernelUtils::cudaLaunchKernel<SequenceGraphKernels::deletions>(SequenceGraphKernels::DELETIONS_DYNAMIC_SMEM_SIZE, SequenceGraphKernels::DELETIONS_MAX_BLOCK_SIZE, pangenomeGraph->getNumNodes(), sync, d_instance, layerIdx);
    }

    __host__ void SequenceGraph::insertions(const Frontier* const frontier, const bool sync) const {
        // Launch insertions kernel
        KernelUtils::cudaLaunchKernel<SequenceGraphKernels::insertions>(SequenceGraphKernels::INSERTIONS_DYNAMIC_SMEM_SIZE, SequenceGraphKernels::INSERTIONS_MAX_BLOCK_SIZE, pangenomeGraph->getNumNodes(), sync, d_instance, frontier->getDeviceInstance());
    }

    __host__ void SequenceGraph::propagations(const ::size_t layerIdx, const Frontier* const frontier, const bool sync) const {
        // Launch propagations kernel
        KernelUtils::cudaLaunchKernel<SequenceGraphKernels::propagations>(SequenceGraphKernels::PROPAGATIONS_DYNAMIC_SMEM_SIZE, SequenceGraphKernels::PROPAGATIONS_MAX_BLOCK_SIZE, frontier->getSize(), sync, d_instance, frontier->getDeviceInstance(), layerIdx);
    }

    __host__ void SequenceGraph::minScore(const bool sync) const {
        // Launch min score kernel
        KernelUtils::cudaLaunchKernel<SequenceGraphKernels::minScore>(SequenceGraphKernels::MIN_SCORE_DYNAMIC_SMEM_SIZE, SequenceGraphKernels::MIN_SCORE_MAX_BLOCK_SIZE, pangenomeGraph->getNumNodes(), sync, d_instance);
    }

    __host__ __device__ SequenceGraph::SequenceGraph(PackedDNASequence* const sequence, PangenomeGraph* const pangenomeGraph, PangenomeGraph* const (& characterGraphs)[NUM_BASES], DoubleBuffer<::cuda::atomic<::uint64_t, ::cuda::thread_scope_device>>* const costsDoubleBuffer, const ::uint64_t score, SequenceGraph* const d_instance) : sequence(sequence), pangenomeGraph(pangenomeGraph), costsDoubleBuffer(costsDoubleBuffer), score({score}), d_instance(d_instance) {
        // Initialize character graphs
        for (::size_t i{0}; i < NUM_BASES; ++i) {
            this->characterGraphs[i] = characterGraphs[i];
        }
    }

    __global__ void SequenceGraphKernels::initialize(const SequenceGraph* sequenceGraph) {
        // Get thread node index
        const auto nodeIdx{::blockIdx.x * ::blockDim.x + ::threadIdx.x};

        // Get pangenome graph
        const auto pangenomeGraph{sequenceGraph->getPangenomeGraph()};

        // Check for thread overflow
        if (const auto numNodes{pangenomeGraph->getNumNodes()}; nodeIdx < numNodes) {
            // Get costs double buffer
            const auto costsDoubleBuffer{sequenceGraph->getCostsDoubleBuffer()};

            // Get current DNA base in the sequence
            const auto sequence{sequenceGraph->getSequence()};
            const auto currentBase{(*sequence)[0]};

            // Check if match
            const auto isMatch{pangenomeGraph->getDNABase(nodeIdx) == currentBase};

            // Compute updated node cost
            const auto updatedCost{(isMatch)? 0 : SequenceGraph::INITIALIZATION_COST};

            // Initialize node cost for the next layer
            costsDoubleBuffer->current()[nodeIdx].store(updatedCost, ::cuda::memory_order_relaxed);
        }
    }

    __global__ void SequenceGraphKernels::substitutions(const SequenceGraph* const sequenceGraph, const ::size_t layerIdx) {
        // Get thread node index
        const auto nodeIdx{::blockIdx.x * ::blockDim.x + ::threadIdx.x};

        // Get pangenome graph
        const auto pangenomeGraph{sequenceGraph->getPangenomeGraph()};

        // Check for thread overflow
        if (const auto numNodes{pangenomeGraph->getNumNodes()}; nodeIdx < numNodes) {
            // Get costs double buffer
            const auto costsDoubleBuffer{sequenceGraph->getCostsDoubleBuffer()};

            // Get neighbors
            const auto neighbors{pangenomeGraph->getNeighbors(nodeIdx)};
            const auto numNeighbors{pangenomeGraph->getNumNeighbors(nodeIdx)};

            // Get node cost in the previous layer
            const auto previousLayerCost{costsDoubleBuffer->alternate()[nodeIdx].load(::cuda::memory_order_relaxed)};

            // Get current DNA base in the sequence
            const auto sequence{sequenceGraph->getSequence()};
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

    __global__ void SequenceGraphKernels::deletions(const SequenceGraph* const sequenceGraph, const ::size_t layerIdx) {
        // Get thread node index
        const auto nodeIdx{::blockIdx.x * ::blockDim.x + ::threadIdx.x};

        // Get pangenome graph
        const auto pangenomeGraph{sequenceGraph->getPangenomeGraph()};

        // Check for thread overflow
        if (const auto numNodes{pangenomeGraph->getNumNodes()}; nodeIdx < numNodes) {
            // Get costs double buffer
            const auto costsDoubleBuffer{sequenceGraph->getCostsDoubleBuffer()};

            // Get node cost in the previous layer
            const auto previousLayerCost{costsDoubleBuffer->alternate()[nodeIdx].load(::cuda::memory_order_relaxed)};

            // Compute updated node cost
            const auto updatedCost{previousLayerCost + SequenceGraph::DELETION_COST};

            // Initialize node cost for the next layer
            costsDoubleBuffer->current()[nodeIdx].store(updatedCost, ::cuda::memory_order_relaxed);
        }
    }

    __global__ void SequenceGraphKernels::insertions(const SequenceGraph* const sequenceGraph, Frontier* const frontier) {
        // Get thread node index
        const auto nodeIdx{::blockIdx.x * ::blockDim.x + ::threadIdx.x};

        // Check for thread overflow
        if (const auto numNodes{sequenceGraph->getPangenomeGraph()->getNumNodes()}; nodeIdx < numNodes) {
            // Insert node in the frontier without queueing
            frontier->insertWithoutQueueing(nodeIdx);

            // Set initial frontier size
            if (nodeIdx == 0) {
                frontier->setSize(numNodes);
            }
        }
    }

    __global__ void SequenceGraphKernels::propagations(const SequenceGraph* sequenceGraph, Frontier* const frontier, const ::size_t layerIdx) {
        // Get thread node index
        const auto threadId{::blockIdx.x * ::blockDim.x + ::threadIdx.x};

        // Get pangenome graph
        const auto pangenomeGraph{sequenceGraph->getPangenomeGraph()};

        // Get current DNA base in the sequence
        const auto sequence{sequenceGraph->getSequence()};
        const auto currentBase{(*sequence)[layerIdx]};

        // Check for thread overflow and if given node is a root in the corresponding character graph for the given layer
        if (const auto frontierSize{frontier->getSize()}; threadId < frontierSize) {
            // Get node index
            const auto nodeIdx{frontier->getNodeIndex(threadId)};

            // Get costs double buffer
            const auto costsDoubleBuffer{sequenceGraph->getCostsDoubleBuffer()};

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
                            frontier->atomicInsertAndGrow(neighborIdx);
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