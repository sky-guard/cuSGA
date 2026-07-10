#include "PangenomeGraph.cuh"

#include <format>
#include <fstream>

#include "KernelUtils.cuh"

namespace cuSGA {
    __host__ PangenomeGraph* PangenomeGraph::createFromFile(const std::string& fileName) {
        // Open pangenome graph file
        std::ifstream file{fileName};
        if (!file.is_open()) {
            throw std::runtime_error{std::format("Unable to open file: {}", fileName)};
        }

        // Read pangenome graph from file
        const auto pangenomeGraph{createFromFile(fileName, file)};

        // Close pangenome graph file
        file.close();

        return pangenomeGraph;
    }

    __host__ PangenomeGraph* PangenomeGraph::createFromFile(const std::string& fileName, std::ifstream& file, std::optional<::size_t> numNodes, std::optional<::size_t> numEdges, std::optional<PackedDNASequence*> baseValues, std::optional<const ::size_t*> columnValues, std::optional<const ::size_t*> rowOffsets) {
        // Read number of nodes from file if missing
        if (!numNodes.has_value()) {
            ::size_t numNodesValue{0};
            if (!(file >> numNodesValue)) {
                throw std::runtime_error{std::format("An error occurred while reading values from file: {}", fileName)};
            }

            // Assign value
            numNodes = numNodesValue;
        }

        // Read number of edges from file if missing
        if (!numEdges.has_value()) {
            ::size_t numEdgesValue{0};
            if (!(file >> numEdgesValue)) {
                throw std::runtime_error{std::format("An error occurred while reading values from file: {}", fileName)};
            }

            // Assign value
            numEdges = numEdgesValue;
        }

        // Read nodes from file if missing
        if (!baseValues.has_value()) {
            // Assign value
            baseValues = PackedDNASequence::createFromFile(fileName, file, numNodes);
        }

        // Read row offsets from file if missing
        if (!rowOffsets.has_value()) {
            // Allocate buffer
            const auto numNodesValue{numNodes.value()};
            const auto rowOffsetsValue{new ::size_t[numNodesValue + 1]{}};

            // Read the row offsets from file
            for (::size_t i{0}; i <= numNodesValue; ++i) {
                if (!(file >> rowOffsetsValue[i])) {
                    throw std::runtime_error{std::format("An error occurred while reading CSR row offsets from file: {}", fileName)};
                }
            }

            // Assign value
            rowOffsets = rowOffsetsValue;
        }

        // Read column values from file if missing
        if (!columnValues.has_value() || !rowOffsets.has_value()) {
            // Allocate buffer
            const auto numEdgesValue{numEdges.value()};
            const auto columnValuesValue{new ::size_t[numEdgesValue]{}};

            // Read the column values from file
            for (::size_t i{0}; i < numEdgesValue; ++i) {
                if (!(file >> columnValuesValue[i])) {
                    throw std::runtime_error{std::format("An error occurred while reading CSR column values from file: {}", fileName)};
                }
            }

            // Assign value
            columnValues = columnValuesValue;
        }

        // Create pangenome graph instance
        const auto pangenomeGraph{new PangenomeGraph{numNodes.value(), numEdges.value(), baseValues.value(), columnValues.value(), rowOffsets.value()}};

        return pangenomeGraph;
    }

    __host__ PangenomeGraph* PangenomeGraph::copyToDevice() {
        // Check if device instance already exists for this pangenome graph
        if (d_instance) {
            return d_instance;
        }

        // Allocate device pangenome graph buffers
        ::size_t* d_columnValues{nullptr};
        ::size_t* d_rowOffsets{nullptr};
        KernelUtils::cudaMalloc(&d_columnValues, numEdges * sizeof(::size_t));
        KernelUtils::cudaMalloc(&d_rowOffsets, (numNodes + 1) * sizeof(::size_t));

        // Copy buffers data from host to device
        const auto d_baseValues{baseValues->copyToDevice()};
        KernelUtils::cudaMemcpy(d_columnValues, columnValues, numEdges * sizeof(::size_t), ::cudaMemcpyHostToDevice);
        KernelUtils::cudaMemcpy(d_rowOffsets, rowOffsets, (numNodes + 1) * sizeof(::size_t), ::cudaMemcpyHostToDevice);

        // Allocate device pangenome graph instance
        PangenomeGraph* d_pangenomeGraph{nullptr};
        KernelUtils::cudaMalloc(&d_pangenomeGraph, sizeof(PangenomeGraph));

        // Create temporary host instance holding the device pointers
        const PangenomeGraph devicePangenomeGraph{numNodes, numEdges, d_baseValues, d_columnValues, d_rowOffsets, d_pangenomeGraph};

        // Update host instance data
        this->d_instance = d_pangenomeGraph;

        // Copy instance data from host to device
        KernelUtils::cudaMemcpy(d_pangenomeGraph, &devicePangenomeGraph, sizeof(PangenomeGraph), ::cudaMemcpyHostToDevice);

        return d_pangenomeGraph;
    }

    __host__ void PangenomeGraph::free(const bool freeSequence) const {
        // Free device memory if device instance is present
        if (d_instance) {
            // Create a temporary host copy of the device instance to get its internal device pointers
            PangenomeGraph devicePangenomeGraph{};
            KernelUtils::cudaMemcpy(&devicePangenomeGraph, d_instance, sizeof(PangenomeGraph), ::cudaMemcpyDeviceToHost);

            // Free device pangenome graph buffers
            if (devicePangenomeGraph.columnValues) {
                KernelUtils::cudaFree(const_cast<::size_t*>(devicePangenomeGraph.columnValues));
            }
            if (devicePangenomeGraph.rowOffsets) {
                KernelUtils::cudaFree(const_cast<::size_t*>(devicePangenomeGraph.rowOffsets));
            }

            // Free device pangenome graph instance
            KernelUtils::cudaFree(d_instance);
        }

        // Free host memory
        if (baseValues && freeSequence) {
            this->baseValues->free();
        }
        delete[] columnValues;
        delete[] rowOffsets;
        delete this;
    }

    __host__ __device__ ::size_t PangenomeGraph::getNumNodes() const {
        // Get number of nodes
        return numNodes;
    }

    __host__ __device__ ::size_t PangenomeGraph::getNumEdges() const {
        // Get number of edges
        return numEdges;
    }

    __host__ __device__ PackedDNASequence* PangenomeGraph::getBaseValues() const {
        // Get base values
        return baseValues;
    }

    __host__ __device__ DNABase PangenomeGraph::getDNABase(const ::size_t nodeIdx) const {
        // Get base from sequence
        const auto base{(*baseValues)[nodeIdx]};

        return base;
    }

    __host__ __device__ const ::size_t *PangenomeGraph::getNeighbors(const ::size_t nodeIdx) const {
        // Get row offset for current node
        const auto rowOffset{rowOffsets[nodeIdx]};

        // Compute address for neighbors
        const auto neighbors{&columnValues[rowOffset]};

        return neighbors;
    }

    __host__ __device__ ::size_t PangenomeGraph::getNumNeighbors(const ::size_t nodeIdx) const {
        // Get row offsets for current node and next in the sequence
        const auto rowOffset{rowOffsets[nodeIdx]};
        const auto nextRowOffset{rowOffsets[nodeIdx + 1]};

        // Compute number of neighbors as the difference between the two
        const auto numNeighbors{nextRowOffset - rowOffset};

        return numNeighbors;
    }

    __host__ __device__ PangenomeGraph* PangenomeGraph::getDeviceInstance() const {
        return d_instance;
    }

    __host__ __device__ PangenomeGraph::PangenomeGraph(const ::size_t numNodes, const ::size_t numEdges, PackedDNASequence* const baseValues, const ::size_t* const columnValues, const ::size_t* const rowOffsets, PangenomeGraph* const d_instance) : numNodes(numNodes), numEdges(numEdges), baseValues(baseValues), columnValues(columnValues), rowOffsets(rowOffsets), d_instance(d_instance) {}
} // cuSGA
