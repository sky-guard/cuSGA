#include "PackedDNASequence.cuh"

#include <format>
#include <fstream>

#include "KernelUtils.cuh"
#include "Utils.cuh"

namespace cuSGA {
    __host__ PackedDNASequence::PackedDNASequence(const ::std::string& fileName, const bool ownsInstance, PackedDNASequence* const pinned_instanceOptional, KernelUtils::BumpPtrAllocator* const allocatorOptional) : PackedDNASequence() {
        // Get allocator
        KernelUtils::BumpPtrAllocator* allocator{allocatorOptional};
        KernelUtils::BumpPtrAllocator allocatorInstance{};
        if (!allocator) {
            allocator = &allocatorInstance;
        }

        // Open file
        auto file{Utils::openFile(fileName)};

        // Read max number of bases from file
        ::size_t maxNumBases{0};
        if (!(file >> maxNumBases)) {
            throw ::std::runtime_error{::std::format("An error occurred while reading values from file: {}", fileName)};
        }

        // Close file
        file.close();

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
        this->numChunks = (maxNumBases + PACKING_FACTOR - 1) / PACKING_FACTOR;
        this->bases = allocator->emplaceReserve<::std::remove_pointer_t<decltype(bases)>>(numChunks);
        this->ownsInstance = ownsInstance;
    }

    __host__ PackedDNASequence::PackedDNASequence(const ::std::string& fileName, ::std::ifstream* const file, const bool ownsInstance, ::std::optional<::size_t> numBasesOptional, PackedDNASequence* const pinned_instanceOptional, KernelUtils::BumpPtrAllocator* const allocatorOptional) : PackedDNASequence(0, 0, nullptr, ownsInstance, pinned_instanceOptional) {
        // Get allocator
        KernelUtils::BumpPtrAllocator* allocator{allocatorOptional};
        KernelUtils::BumpPtrAllocator allocatorInstance{};
        if (!allocator) {
            allocator = &allocatorInstance;
        }

        // Read number of bases from file if missing
        if (!numBasesOptional.has_value()) {
            ::size_t numBases{0};
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
        this->numChunks = (numBases + PACKING_FACTOR - 1) / PACKING_FACTOR;
        this->bases = allocator->emplaceReserve<::std::remove_pointer_t<decltype(bases)>>(numChunks);

        // Read base values from file
        for (::size_t i{0}; i < numBases; ++i) {
            // Read character from file
            char c{'\0'};
            if (!(*file >> c)) {
                throw ::std::runtime_error{::std::format("An error occurred while reading values from file: {}", fileName)};
            }

            // Convert character to DNA base
            const auto base{DNABaseConversion::charToDNABase(c)};

            // Get chunk index and bit offset
            const auto chunkIdx{i / PACKING_FACTOR};
            const auto bitOffset{(i % PACKING_FACTOR) * DNA_BASE_BIT_SIZE};

            // Pack into the pack_t
            bases[chunkIdx] |= static_cast<pack_t>(base) << bitOffset;
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
            growBuffers(allocator, numChunks * PACKING_FACTOR);
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
        const auto d_bases{allocator->cudaEmplaceCopy<::std::remove_pointer_t<decltype(bases)>>(bases, ::cudaMemcpyHostToDevice, numChunks, false, cudaStreamDefault)};

        // Create temporary host instance holding the device pointers
        const PackedDNASequence d_sequence{numBases, numChunks, d_bases, ownsInstance, pinned_instance, d_instance};

        // Emplace instance
        if (ownsInstance) {
            *pinned_instance = d_sequence;
            KernelUtils::cudaMemcpyAsync(d_instance, pinned_instance, sizeof(PackedDNASequence), ::cudaMemcpyHostToDevice, cudaStreamDefault);
        }

        return d_sequence;
    }

    __host__ void PackedDNASequence::free() const {
        // Free device memory if present
        if (d_instance) {
            if (ownsInstance) {
                KernelUtils::cudaFreeAsync(d_instance, cudaStreamDefault);
            }
            else {
                KernelUtils::cudaFreeAsync(getBuffersRoot(), cudaStreamDefault);
            }
        }

        // Free host memory
        if (ownsInstance) {
            KernelUtils::cudaFreeHost(pinned_instance);
        }
        else {
            KernelUtils::cudaFreeHost(getBuffersRoot());
        }
    }

    __host__ bool PackedDNASequence::readFromFile(const ::std::string& fileName, ::std::ifstream* const file) const {
        // Read number of bases from file
        ::size_t numBases{0};
        if (!(*file >> numBases)) {
            // Check EOF
            if (file->eof()) {
                return false;
            }

            // Throw exception for read error
            throw ::std::runtime_error{::std::format("An error occurred while reading values from file: {}", fileName)};
        }

        // Read base values from file
        for (::size_t i{0}; i < numBases; ++i) {
            // Read character from file
            char c{'\0'};
            if (!(*file >> c)) {
                throw ::std::runtime_error{::std::format("An error occurred while reading values from file: {}", fileName)};
            }

            // Convert character to DNA base
            const auto base{DNABaseConversion::charToDNABase(c)};

            // Get chunk index and bit offset
            const auto chunkIdx{i / PACKING_FACTOR};
            const auto bitOffset{(i % PACKING_FACTOR) * DNA_BASE_BIT_SIZE};

            // Check if chunk needs to be cleared before writing
            if (i % PACKING_FACTOR == 0) {
                bases[chunkIdx] = 0;
            }

            // Pack into the pack_t
            bases[chunkIdx] |= static_cast<pack_t>(base) << bitOffset;
        }

        // Update device instance
        if (d_instance) {
            // Update number of bases
            pinned_instance->numBases = numBases;

            // Update device buffer
            const auto newNumChunks{(numBases + PACKING_FACTOR - 1) / PACKING_FACTOR};
            KernelUtils::cudaMemcpyAsync(pinned_instance->bases, bases, newNumChunks * sizeof(bases[0]), ::cudaMemcpyHostToDevice, cudaStreamDefault);

            // Update device instance
            KernelUtils::cudaMemcpyAsync(d_instance, pinned_instance, sizeof(PackedDNASequence), ::cudaMemcpyHostToDevice, cudaStreamDefault);
        }

        return true;
    }
} // cuSGA