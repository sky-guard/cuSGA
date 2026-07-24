#include "PackedDNASequence.cuh"

#include <format>
#include <fstream>

#include "KernelUtils.cuh"
#include "SequenceGraph.cuh"
#include "Utils.cuh"

namespace cuSGA {
    __host__ PackedDNASequence::PackedDNASequence(const ::std::string& fileName, const bool ownsInstance, PackedDNASequence* const pinned_instanceOptional, KernelUtils::BumpPtrAllocator* const allocatorOptional) : PackedDNASequence{} {
        // Get allocator
        KernelUtils::BumpPtrAllocator* allocator{allocatorOptional};
        KernelUtils::BumpPtrAllocator allocatorInstance{};
        if (!allocator) {
            allocator = &allocatorInstance;
        }

        // Open file
        auto file{Utils::openFile(fileName)};

        // Read and skip number of scores from file
        if (scoreSize_t numScores{0}; !(file >> numScores)) {
            throw ::std::runtime_error{::std::format("An error occurred while reading values from file: {}", fileName)};
        }

        // Read max number of bases from file
        sequenceSize_t maxNumBases{0};
        if (!(file >> maxNumBases)) {
            throw ::std::runtime_error{::std::format("An error occurred while reading values from file: {}", fileName)};
        }

        // Grow allocator
        if (ownsInstance) {
            allocator->emplaceReserve<PackedDNASequence>();
            growBuffers(allocator, maxNumBases);
        }

        // Initialize allocator
        if (ownsInstance) {
            allocator->initHostPinnedMem();
        }

        // Emplace buffers
        if (ownsInstance) {
            this->pinned_instance = allocator->emplaceReserve<PackedDNASequence>();
        }
        else {
            this->pinned_instance = pinned_instanceOptional;
        }
        this->numChunks = (maxNumBases + PACKING_FACTOR - 1) >> PACK_SHIFT;
        this->bases = allocator->emplaceReserve<::std::remove_reference_t<decltype(bases[0])>>(numChunks);
        this->ownsInstance = ownsInstance;

        // Close file
        file.close();
    }

    __host__ PackedDNASequence::PackedDNASequence(const ::std::string& fileName, ::std::ifstream* const file, const bool ownsInstance, ::std::optional<sequenceSize_t> numBasesOptional, PackedDNASequence* const pinned_instanceOptional, KernelUtils::BumpPtrAllocator* const allocatorOptional) : PackedDNASequence{0, 0, nullptr, ownsInstance, pinned_instanceOptional} {
        // Get allocator
        KernelUtils::BumpPtrAllocator* allocator{allocatorOptional};
        KernelUtils::BumpPtrAllocator allocatorInstance{};
        if (!allocator) {
            allocator = &allocatorInstance;
        }

        // Read number of bases from file if missing
        if (!numBasesOptional.has_value()) {
            sequenceSize_t numBases{0};
            if (!(*file >> numBases)) {
                throw ::std::runtime_error{::std::format("An error occurred while reading values from file: {}", fileName)};
            }
            numBasesOptional = numBases;
        }

        // Grow allocator
        if (ownsInstance) {
            allocator->emplaceReserve<PackedDNASequence>();
            growBuffers(allocator, numBasesOptional.value());
        }

        // Initialize allocator
        if (ownsInstance) {
            allocator->initHostPinnedMem();
        }

        // Emplace buffers
        if (ownsInstance) {
            this->pinned_instance = pinned_instanceOptional;
        }
        this->numBases = numBasesOptional.value();
        this->numChunks = (numBases + PACKING_FACTOR - 1) >> PACK_SHIFT;
        this->bases = allocator->emplaceReserve<::std::remove_reference_t<decltype(bases[0])>>(numChunks);

        // Read base values from file
        for (sequenceSize_t sequenceIdx{0}; sequenceIdx < numBases; ++sequenceIdx) {
            // Read character from file
            char c{'\0'};
            if (!(*file >> c)) {
                throw ::std::runtime_error{::std::format("An error occurred while reading values from file: {}", fileName)};
            }

            // Convert character to DNA base
            const auto base{DNABaseConversion::charToDNABase(c)};

            // Get chunk index and bit offset
            const auto chunkIdx{sequenceIdx / PACKING_FACTOR};
            const auto bitOffset{(sequenceIdx % PACKING_FACTOR) * BIT_SIZE};

            // Pack into the pack_t
            bases[chunkIdx] |= (static_cast<sequencePack_t>(base) << bitOffset);
        }
    }

    __host__ PackedDNASequence PackedDNASequence::copyToDevice(PackedDNASequence* const d_instanceOptional, KernelUtils::BumpPtrAllocator* allocatorOptional) {
        // Check if device instance already exists for this sequence
        if (d_instance) {
            throw ::std::runtime_error{"Device instance already exists for this Packed Sequence!"};
        }

        // Get allocator
        KernelUtils::BumpPtrAllocator* allocator{allocatorOptional};
        KernelUtils::BumpPtrAllocator allocatorInstance{};
        if (!allocator) {
            allocator = &allocatorInstance;
        }

        // Grow allocator
        if (ownsInstance) {
            allocator->grow<PackedDNASequence>();
            growBuffers(allocator, numChunks << PACK_SHIFT);
        }

        // Initialize allocator
        if (ownsInstance) {
            allocator->initCudaGMem();
        }

        // Reserve instance
        if (ownsInstance) {
            this->d_instance = allocator->emplaceReserve<PackedDNASequence>();
        }
        else {
            this->d_instance = d_instanceOptional;
        }

        // Emplace buffers
        const auto d_bases{allocator->cudaEmplaceCopy<::std::remove_reference_t<decltype(bases[0])>>(bases, ::cudaMemcpyHostToDevice, numChunks, false, cudaStreamDefault)};

        // Create temporary host instance holding the device pointers
        const PackedDNASequence d_sequence{numBases, numChunks, d_bases, ownsInstance, pinned_instance, d_instance};

        // Emplace instance
        if (ownsInstance) {
            *pinned_instance = d_sequence;
            CUDA_CHECK(::cudaMemcpyAsync(d_instance, pinned_instance, sizeof(PackedDNASequence), ::cudaMemcpyHostToDevice, cudaStreamDefault));
        }

        return d_sequence;
    }

    __host__ void PackedDNASequence::free() const {
        // Free device memory if present
        if (d_instance) {
            if (ownsInstance) {
                CUDA_CHECK(::cudaFreeAsync(d_instance, cudaStreamDefault));
            }
            else {
                CUDA_CHECK(::cudaFreeAsync(getBuffersRoot(), cudaStreamDefault));
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

    __host__ bool PackedDNASequence::readFromFile(const ::std::string& fileName, ::std::ifstream* const file) const {
        // Read number of bases from file
        sequenceSize_t numBases{0};
        if (!(*file >> numBases)) {
            // Check EOF
            if (file->eof()) {
                return false;
            }

            // Throw exception for read error
            throw ::std::runtime_error{::std::format("An error occurred while reading values from file: {}", fileName)};
        }

        // Read base values from file
        for (sequenceSize_t sequenceIdx{0}; sequenceIdx < numBases; ++sequenceIdx) {
            // Read character from file
            char c{'\0'};
            if (!(*file >> c)) {
                throw ::std::runtime_error{::std::format("An error occurred while reading values from file: {}", fileName)};
            }

            // Convert character to DNA base
            const auto base{DNABaseConversion::charToDNABase(c)};

            // Get chunk index and bit offset
            const auto chunkIdx{sequenceIdx >> PACK_SHIFT};
            const auto bitOffset{(sequenceIdx & (PACKING_FACTOR - 1)) * BIT_SIZE};

            // Check if chunk needs to be cleared before writing
            if ((sequenceIdx & (PACKING_FACTOR - 1)) == 0) {
                bases[chunkIdx] = 0;
            }

            // Pack into the pack_t
            bases[chunkIdx] |= (static_cast<sequencePack_t>(base) << bitOffset);
        }

        // Update device instance
        if (d_instance) {
            // Update number of bases
            pinned_instance->numBases = numBases;

            // Update device buffer
            const auto newNumChunks{(numBases + PACKING_FACTOR - 1) >> PACK_SHIFT};
            CUDA_CHECK(::cudaMemcpyAsync(pinned_instance->bases, bases, newNumChunks * sizeof(bases[0]), ::cudaMemcpyHostToDevice, cudaStreamDefault));

            // Update device instance
            CUDA_CHECK(::cudaMemcpyAsync(d_instance, pinned_instance, sizeof(PackedDNASequence), ::cudaMemcpyHostToDevice, cudaStreamDefault));
        }

        return true;
    }
} // cuSGA