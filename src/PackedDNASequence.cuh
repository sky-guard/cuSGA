#ifndef CUSGA_PACKEDDNASEQUENCE_CUH
#define CUSGA_PACKEDDNASEQUENCE_CUH
#include <cstdint>
#include <optional>
#include <string>

#include "DNABase.cuh"
#include "KernelUtils.cuh"
#include "Utils.cuh"

namespace cuSGA {
    // Define pack type
    using sequencePack_t = ::uint32_t;

    // Define sequence size type
    using sequenceSize_t = targetSize_t;

    // Packed DNA sequence
    class PackedDNASequence {
    public:
        // Packed sequence related constants
        static constexpr ::uint8_t BIT_SIZE{2};
        static constexpr sequencePack_t BITMASK{(1u << BIT_SIZE) - 1};
        static constexpr ::uint8_t PACKING_FACTOR{(sizeof(sequencePack_t) * Utils::BYTE_SIZE) / BIT_SIZE};
        static constexpr ::uint8_t PACK_SHIFT{::std::countr_zero(PACKING_FACTOR)};

        // Grow allocator using the expected buffers size
        __host__ __device__ __forceinline__ static void growBuffers(KernelUtils::BumpPtrAllocator* const allocator, const sequenceSize_t numBases) {
            // Grow size for bases
            const auto numChunks{(numBases + PACKING_FACTOR - 1) >> PACK_SHIFT};
            allocator->grow<::std::remove_reference_t<decltype(bases[0])>>(numChunks);
        }

        // Default constructor
        PackedDNASequence() = default;
        // Parameterized constructor
        __host__ PackedDNASequence(const ::std::string& fileName, bool ownsInstance, PackedDNASequence* pinned_instanceOptional = nullptr, KernelUtils::BumpPtrAllocator* allocatorOptional = nullptr);
        // Parameterized constructor
        __host__ PackedDNASequence(const ::std::string& fileName, ::std::ifstream* file, bool ownsInstance, ::std::optional<sequenceSize_t> numBasesOptional = ::std::nullopt, PackedDNASequence* pinned_instanceOptional = nullptr, KernelUtils::BumpPtrAllocator* allocatorOptional = nullptr);

        // Parameterized constructor
        __host__ __device__ __forceinline__ PackedDNASequence(const sequenceSize_t numBases, const sequenceSize_t numChunks, sequencePack_t* const bases, const bool ownsInstance, PackedDNASequence* const pinned_instance = nullptr, PackedDNASequence* const d_instance = nullptr) : numBases{numBases}, numChunks{numChunks}, bases{bases}, ownsInstance{ownsInstance}, pinned_instance{pinned_instance}, d_instance{d_instance} {}

        // Copy constructor
        PackedDNASequence(const PackedDNASequence& other) = default;
        // Move constructor
        PackedDNASequence(PackedDNASequence&& other) = default;
        // Copy assignment
        PackedDNASequence& operator=(const PackedDNASequence& other) = default;
        // Move assignment
        PackedDNASequence& operator=(PackedDNASequence&& other) = default;
        // Destructor
        ~PackedDNASequence() = default;

        // Move packed sequence to device
        __host__ PackedDNASequence copyToDevice(PackedDNASequence* d_instanceOptional = nullptr, KernelUtils::BumpPtrAllocator* allocatorOptional = nullptr);
        // Free packed sequence
        __host__ void free() const;

        // Get number of bases
        __host__ __device__ __forceinline__ sequenceSize_t getNumBases() const {
            return numBases;
        }

        // Get number of chunks
        __host__ __device__ __forceinline__ sequenceSize_t getNumChunks() const {
            return numChunks;
        }

        // Get DNA base at a given index
        __host__ __device__ __forceinline__ DNABase operator[](const size_t idx) const {
            // Get chunk index and bit offset from node index
            const auto chunkIdx{idx >> PACK_SHIFT};
            const auto bitOffset{(idx & (PACKING_FACTOR - 1)) * BIT_SIZE};

            // Get chunk, shift bits and extract using mask
            const auto base{static_cast<DNABase>((bases[chunkIdx] >> bitOffset) & BITMASK)};

            return base;
        }

        // Get pinned instance
        __host__ __device__ __forceinline__ PackedDNASequence* getPinnedInstance() const {
            return pinned_instance;
        }

        // Get device instance
        __host__ __device__ __forceinline__ PackedDNASequence* getDeviceInstance() const {
            return d_instance;
        }

        // Get buffer root
        __host__ __device__ __forceinline__ void* getBuffersRoot() const {
            return bases;
        }

        // Read packed sequence from opened file
        __host__ bool readFromFile(const ::std::string& fileName, ::std::ifstream* file);

        // Shuffle object from the given lane, with the given mask
        __device__ __forceinline__ void shuffle_sync(const unsigned mask, const int srcLaneIdx) {
            this->numBases = ::__shfl_sync(mask, numBases, srcLaneIdx);
            this->numChunks = ::__shfl_sync(mask, numChunks, srcLaneIdx);
            this->bases = reinterpret_cast<decltype(bases)>(::__shfl_sync(mask, reinterpret_cast<unsigned long long>(bases), srcLaneIdx));
            this->ownsInstance = ::__shfl_sync(mask, ownsInstance, srcLaneIdx);
            this->pinned_instance = reinterpret_cast<decltype(pinned_instance)>(::__shfl_sync(mask, reinterpret_cast<unsigned long long>(pinned_instance), srcLaneIdx));
            this->d_instance = reinterpret_cast<decltype(d_instance)>(::__shfl_sync(mask, reinterpret_cast<unsigned long long>(d_instance), srcLaneIdx));
        }

    private:
        // Bit packed sequence implementation
        // NOTE: Uses pinned memory and linearized memory layout on the device memory
        sequenceSize_t numBases{0};
        sequenceSize_t numChunks{0};
        sequencePack_t* bases{nullptr};
        bool ownsInstance{false};
        PackedDNASequence* pinned_instance{nullptr};
        PackedDNASequence* d_instance{nullptr};
    };
} // cuSGA

#endif //CUSGA_PACKEDDNASEQUENCE_CUH
