#include "PackedDNASequence.cuh"

#include <format>
#include <fstream>

#include "KernelUtils.cuh"

namespace cuSGA {
    __host__ PackedDNASequence* PackedDNASequence::create() {
        // Create sequence instance
        const auto sequence{new PackedDNASequence{}};

        return sequence;
    }

    __host__ PackedDNASequence* PackedDNASequence::createFromFile(const ::std::string& fileName) {
        // Open sequence file
        ::std::ifstream file{fileName};
        if (!file.is_open()) {
            throw ::std::runtime_error{::std::format("Unable to open file: {}", fileName)};
        }

        // Read sequence from file
        const auto sequence{createFromFile(fileName, file)};

        // Close sequence file
        file.close();

        return sequence;
    }

    __host__ PackedDNASequence* PackedDNASequence::createFromFile(const ::std::string& fileName, ::std::ifstream& file, ::std::optional<::size_t> numBases, ::std::optional<::uint64_t*> bases) {
        // Read number of bases from file if missing
        if (!numBases.has_value()) {
            ::size_t numBasesValue{0};
            if (!(file >> numBasesValue)) {
                throw ::std::runtime_error{::std::format("An error occurred while reading values from file: {}", fileName)};
            }

            // Assign value
            numBases = numBasesValue;
        }

        // Read bases from file if missing
        const auto numChunks{(numBases.value() + PACKING_FACTOR - 1) / PACKING_FACTOR};
        if (!bases.has_value()) {
            // Allocate host sequence buffer
            const auto basesValue{new ::uint64_t[numChunks]{}};

            // Read base values from file
            const auto numBasesValue{numBases.value()};
            for (::size_t i{0}; i < numBasesValue; ++i) {
                // Read character from file
                char c{'\0'};
                if (!(file >> c)) {
                    throw ::std::runtime_error{::std::format("An error occurred while reading values from file: {}", fileName)};
                }

                // Convert character to DNA base
                const auto base{DNABaseConversion::charToDNABase(c)};

                // Get chunk index and bit offset
                const auto chunkIdx{i / PACKING_FACTOR};
                const auto bitOffset{(i % PACKING_FACTOR) * DNA_BASE_BIT_SIZE};

                // Pack into the 64-bit integer
                basesValue[chunkIdx] |= static_cast<::uint64_t>(base) << bitOffset;
            }

            // Assign value
            bases = basesValue;
        }

        // Create sequence instance
        const auto sequence{new PackedDNASequence{numBases.value(), numChunks, bases.value()}};

        return sequence;
    }

    __host__ PackedDNASequence* PackedDNASequence::copyToDevice() {
        // Check if device instance already exists for this sequence
        if (d_instance) {
            return d_instance;
        }

        // Allocate device sequence buffer
        ::uint64_t* d_bases{nullptr};
        KernelUtils::cudaMalloc(&d_bases, numChunks * sizeof(::uint64_t));

        // Copy buffer data from host to device
        KernelUtils::cudaMemcpy(d_bases, bases, numChunks * sizeof(::uint64_t), ::cudaMemcpyHostToDevice);

        // Allocate device sequence instance
        PackedDNASequence* d_sequence{nullptr};
        KernelUtils::cudaMalloc(&d_sequence, sizeof(PackedDNASequence));

        // Create temporary host instance holding the device pointers
        const PackedDNASequence deviceSequence{numBases, numChunks, d_bases, d_sequence};

        // Update host instance data
        this->d_instance = d_sequence;

        // Copy instance data from host to device
        KernelUtils::cudaMemcpy(d_sequence, &deviceSequence, sizeof(PackedDNASequence), ::cudaMemcpyHostToDevice);

        return d_sequence;
    }

    __host__ void PackedDNASequence::free() const {
        // Free device memory if device instance is present
        if (d_instance) {
            // Create a temporary host copy of the device instance to get its internal device pointers
            PackedDNASequence deviceSequence{};
            KernelUtils::cudaMemcpy(&deviceSequence, d_instance, sizeof(PackedDNASequence), ::cudaMemcpyDeviceToHost);

            // Free device sequence buffer
            if (deviceSequence.bases) {
                KernelUtils::cudaFree(deviceSequence.bases);
            }

            // Free device graph instance
            KernelUtils::cudaFree(d_instance);
        }

        // Free host memory
        delete[] bases;
        delete this;
    }

    __host__ __device__ ::size_t PackedDNASequence::getNumBases() const {
        return numBases;
    }

    __host__ __device__ ::size_t PackedDNASequence::getNumChunks() const {
        return numChunks;
    }

    __host__ __device__ DNABase PackedDNASequence::operator[](const size_t idx) const {
        // Get chunk index and bit offset from node index
        const auto chunkIdx{idx / PACKING_FACTOR};
        const auto bitOffset{(idx % PACKING_FACTOR) * DNA_BASE_BIT_SIZE};

        // Get packed bases chunk
        const auto chunk{bases[chunkIdx]};

        // Shift and mask to get the 2 LSBs
        const auto baseBits{(chunk >> bitOffset) & DNA_BASE_BITMASK};

        // Cast bits to DNA base
        const auto base{static_cast<DNABase>(baseBits)};

        return base;
    }

    __host__ __device__ PackedDNASequence* PackedDNASequence::getDeviceInstance() const {
        return d_instance;
    }

    __host__ bool PackedDNASequence::readFromFile(const ::std::string& fileName, ::std::ifstream& file) {
        // Read number of bases from file
        if (!(file >> this->numBases)) {
            // Check EOF
            if (file.eof()) {
                return false;
            }

            // Throw exception for read error
            throw ::std::runtime_error{::std::format("An error occurred while reading values from file: {}", fileName)};
        }

        // Get new number of chunks
        const auto numChunks{(numBases + PACKING_FACTOR - 1) / PACKING_FACTOR};

        // Check if buffer needs resizing
        const auto needsResizing{this->numChunks < numChunks};
        if (needsResizing) {
            // Update number of chunks
            this->numChunks = numChunks;

            // Update bases
            delete[] this->bases;
            this->bases = new ::uint64_t[numChunks]{};
        }

        // Read base values from file
        for (::size_t i{0}; i < numBases; ++i) {
            // Read character from file
            char c{'\0'};
            if (!(file >> c)) {
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

            // Pack into the 64-bit integer
            bases[chunkIdx] |= static_cast<::uint64_t>(base) << bitOffset;
        }

        // Update device instance
        if (d_instance) {
            // Create a temporary host copy of the device instance to get its internal device pointers
            PackedDNASequence deviceSequence{};
            KernelUtils::cudaMemcpy(&deviceSequence, d_instance, sizeof(PackedDNASequence), ::cudaMemcpyDeviceToHost);

            // Update number of bases on device
            deviceSequence.numBases = numBases;

            // Resize device instance if necessary
            if (needsResizing) {
                // Update number of chunks on device
                deviceSequence.numChunks = numChunks;

                // Free old buffer
                KernelUtils::cudaFree(deviceSequence.bases);

                // Allocate new buffer
                KernelUtils::cudaMalloc(&deviceSequence.bases, numChunks * sizeof(::uint64_t));
            }

            // Update buffer
            KernelUtils::cudaMemcpy(deviceSequence.bases, bases, numChunks * sizeof(::uint64_t), ::cudaMemcpyHostToDevice);

            // Update instance
            KernelUtils::cudaMemcpy(d_instance, &deviceSequence, sizeof(PackedDNASequence), ::cudaMemcpyHostToDevice);
        }

        return true;
    }

    __host__ __device__ PackedDNASequence::PackedDNASequence(const ::size_t numBases, const ::size_t numChunks, ::uint64_t* const bases, PackedDNASequence* const d_instance) : numBases(numBases), numChunks(numChunks), bases(bases), d_instance(d_instance) {}
} // cuSGA