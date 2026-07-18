#include "PangenomeGraph.cuh"

#include <format>
#include <fstream>

#include "KernelUtils.cuh"

namespace cuSGA {
    __host__ PangenomeGraph::PangenomeGraph(const ::std::string& fileName, bool ownsInstance, PangenomeGraph* pinned_instanceOptional, KernelUtils::BumpPtrAllocator* allocatorOptional) : PangenomeGraph(0, PackedDNASequence{}, nullptr, nullptr, ownsInstance, pinned_instanceOptional) {
        // Get allocator
        KernelUtils::BumpPtrAllocator* allocator{allocatorOptional};
        KernelUtils::BumpPtrAllocator allocatorInstance{};
        if (!allocator) {
            allocator = &allocatorInstance;
        }

        // Open file
        auto file{Utils::openFile(fileName)};

        // Read number of nodes from file
        ::size_t numNodes{0};
        if (!(file >> numNodes)) {
            throw ::std::runtime_error{::std::format("An error occurred while reading values from file: {}", fileName)};
        }

        // Read number of edges from file
        ::size_t numEdges{0};
        if (!(file >> numEdges)) {
            throw ::std::runtime_error{::std::format("An error occurred while reading values from file: {}", fileName)};
        }

        // Close file
        file.close();

        // Grow allocator
        if (ownsInstance) {
            allocator->emplaceReserve<PangenomeGraph>();
            growBuffers(allocator, numNodes, numEdges);
        }

        // Initialize allocator
        if (ownsInstance) {
            allocator->initHostPinnedMem();
        }

        // Emplace buffers
        if (ownsInstance) {
            this->pinned_instance = allocator->emplaceReserve<PangenomeGraph>();
        }
        this->numEdges = numEdges;
        this->baseValues = PackedDNASequence{fileName, &file, false, numNodes, &pinned_instance->baseValues, allocator};
        this->columnValues = allocator->emplaceReserve<::std::remove_pointer_t<decltype(columnValues)>>(numEdges);
        this->rowOffsets = allocator->emplaceReserve<::std::remove_pointer_t<decltype(rowOffsets)>>(numNodes + 1);
    }

    __host__ PangenomeGraph::PangenomeGraph(const ::std::string& fileName, ::std::ifstream* file, const bool ownsInstance, ::std::optional<::size_t> numNodesOptional, ::std::optional<::size_t> numEdgesOptional, PangenomeGraph* pinned_instanceOptional, KernelUtils::BumpPtrAllocator* allocatorOptional) : PangenomeGraph(0, PackedDNASequence{}, nullptr, nullptr, ownsInstance, pinned_instanceOptional) {
        // Get allocator
        KernelUtils::BumpPtrAllocator* allocator{allocatorOptional};
        KernelUtils::BumpPtrAllocator allocatorInstance{};
        if (!allocator) {
            allocator = &allocatorInstance;
        }

        // Read number of nodes from file if missing
        if (!numNodesOptional.has_value()) {
            ::size_t numNodes{0};
            if (!(*file >> numNodes)) {
                throw ::std::runtime_error{::std::format("An error occurred while reading values from file: {}", fileName)};
            }
            numNodesOptional = numNodes;
        }

        // Read number of edges from file if missing
        if (!numEdgesOptional.has_value()) {
            ::size_t numEdges{0};
            if (!(*file >> numEdges)) {
                throw ::std::runtime_error{::std::format("An error occurred while reading values from file: {}", fileName)};
            }
            numEdgesOptional = numEdges;
        }

        // Grow allocator
        if (ownsInstance) {
            allocator->emplaceReserve<PangenomeGraph>();
            growBuffers(allocator, numNodesOptional.value(), numNodesOptional.value());
        }

        // Initialize allocator
        if (ownsInstance) {
            allocator->initHostPinnedMem();
        }

        // Emplace buffers
        if (ownsInstance) {
            this->pinned_instance = allocator->emplaceReserve<PangenomeGraph>();
        }
        this->numEdges = numEdgesOptional.value();
        this->baseValues = PackedDNASequence{fileName, file, false, numNodesOptional.value(), &pinned_instance->baseValues, allocator};
        this->columnValues = allocator->emplaceReserve<::std::remove_pointer_t<decltype(columnValues)>>(numEdges);
        this->rowOffsets = allocator->emplaceReserve<::std::remove_pointer_t<decltype(rowOffsets)>>(numNodesOptional.value() + 1);

        // Read row offsets from file
        for (::size_t i{0}; i <= numNodesOptional.value(); ++i) {
            if (!(*file >> rowOffsets[i])) {
                throw ::std::runtime_error{::std::format("An error occurred while reading CSR row offsets from file: {}", fileName)};
            }
        }

        // Read column values from file
        for (::size_t i{0}; i < numEdges; ++i) {
            if (!(*file >> columnValues[i])) {
                throw ::std::runtime_error{::std::format("An error occurred while reading CSR column values from file: {}", fileName)};
            }
        }
    }

    __host__ PangenomeGraph* PangenomeGraph::copyToDevice(PangenomeGraph* d_instanceOptional, KernelUtils::BumpPtrAllocator* allocatorOptional) {
        // Check if device instance already exists for this pangenome graph
        if (d_instance) {
            throw ::std::runtime_error{"Device instance already exists for this Pangenome Graph!"};
        }

        // Get allocator
        KernelUtils::BumpPtrAllocator* allocator{allocatorOptional};
        KernelUtils::BumpPtrAllocator allocatorInstance{};
        if (!allocator) {
            allocator = &allocatorInstance;
        }

        // Grow allocator
        if (ownsInstance) {
            allocator->grow<PangenomeGraph>();
            growBuffers(allocator, baseValues.getNumBases(), numEdges);
        }

        // Initialize allocator
        if (ownsInstance) {
            allocator->initCudaGMem();
        }

        // Reserve instance
        if (ownsInstance) {
            this->d_instance = allocator->emplaceReserve<PangenomeGraph>();
        }
        else {
            this->d_instance = d_instanceOptional;
        }

        // Emplace buffers
        const auto d_baseValues{baseValues.copyToDevice(&d_instance->baseValues, allocator)};
        const auto d_columnValues{allocator->cudaEmplaceCopy<::std::remove_pointer_t<decltype(columnValues)>>(columnValues, ::cudaMemcpyHostToDevice, numEdges, false, cudaStreamDefault)};
        const auto d_rowOffsets{allocator->cudaEmplaceCopy<::std::remove_pointer_t<decltype(rowOffsets)>>(rowOffsets, ::cudaMemcpyHostToDevice, baseValues.getNumBases() + 1, false, cudaStreamDefault)};

        // Create temporary host instance holding the device pointers
        const PangenomeGraph d_pangenomeGraph{baseValues.getNumBases(), numEdges, d_baseValues, d_columnValues, d_rowOffsets, d_pangenomeGraph};

        // Emplace instance
        if (ownsInstance) {
            *pinned_instance = d_pangenomeGraph;
            CUDA_CHECK(::cudaMemcpyAsync(d_instance, pinned_instance, sizeof(PangenomeGraph), ::cudaMemcpyHostToDevice, cudaStreamDefault));
        }

        return d_pangenomeGraph;
    }

    __host__ void PangenomeGraph::free() const {
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
} // cuSGA
