#ifndef CUSGA_PACKEDDNASEQUENCE_CUH
#define CUSGA_PACKEDDNASEQUENCE_CUH
#include <cstdint>
#include <optional>
#include <string>

#include "DNABase.cuh"
#include "KernelUtils.cuh"
#include "Utils.cuh"

namespace cuSGA {
    class PackedDNASequence {
    public:
        // Define packing type
        using pack_t = ::uint64_t;

        // Packed sequence related constants
        static constexpr ::size_t DNA_BASE_BIT_SIZE{2};
        static constexpr pack_t DNA_BASE_BITMASK{(1 << DNA_BASE_BIT_SIZE) - 1};
        static constexpr ::size_t PACKING_FACTOR{(sizeof(pack_t) * Utils::BYTE_SIZE) / DNA_BASE_BIT_SIZE};

        // Grow allocator using the expected buffers size
        __host__ __device__ __forceinline__ static void growBuffers(KernelUtils::BumpPtrAllocator* const allocator, const ::size_t numBases) {
            // Grow size for bases
            const auto numChunks{(numBases + PACKING_FACTOR - 1) / PACKING_FACTOR};
            allocator->grow<::std::remove_pointer_t<decltype(bases)>>(numChunks);
        }

        // Default constructor
        PackedDNASequence() = default;
        // Parameterized constructor
        __host__ PackedDNASequence(const ::std::string& fileName, bool ownsInstance, PackedDNASequence* pinned_instanceOptional = nullptr, KernelUtils::BumpPtrAllocator* allocatorOptional = nullptr);
        // Parameterized constructor
        __host__ PackedDNASequence(const ::std::string& fileName, ::std::ifstream* file, bool ownsInstance, ::std::optional<::size_t> numBasesOptional = ::std::nullopt, PackedDNASequence* pinned_instanceOptional = nullptr, KernelUtils::BumpPtrAllocator* allocatorOptional = nullptr);
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
        __host__ __device__ __forceinline__::size_t getNumBases() const {
            return numBases;
        }

        // Get number of chunks
        __host__ __device__ __forceinline__ ::size_t getNumChunks() const {
            return numChunks;
        }

        // Get DNA base at a given index
        __host__ __device__ __forceinline__ DNABase operator[](const size_t idx) const {
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
        __host__ bool readFromFile(const ::std::string& fileName, ::std::ifstream* file) const;

    private:
        // Bit packed sequence implementation
        // NOTE: Uses pinned memory and linearized memory layout on the device memory
        ::size_t numBases{0};
        ::size_t numChunks{0};
        pack_t* bases{nullptr};
        bool ownsInstance{false};
        PackedDNASequence* pinned_instance{nullptr};
        PackedDNASequence* d_instance{nullptr};

        // Parameterized constructor
        __host__ __device__ __forceinline__ PackedDNASequence(const ::size_t numBases, const ::size_t numChunks, pack_t* const bases, const bool ownsInstance, PackedDNASequence* const pinned_instance = nullptr, PackedDNASequence* const d_instance = nullptr) : numBases(numBases), numChunks(numChunks), bases(bases), ownsInstance(ownsInstance), pinned_instance(pinned_instance), d_instance(d_instance) {}
    };
} // cuSGA

#endif //CUSGA_PACKEDDNASEQUENCE_CUH
