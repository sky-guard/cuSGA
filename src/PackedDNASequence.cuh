#ifndef CUSGA_PACKEDDNASEQUENCE_CUH
#define CUSGA_PACKEDDNASEQUENCE_CUH
#include <cstdint>
#include <optional>
#include <string>

#include "DNABase.cuh"

namespace cuSGA {
    class PackedDNASequence {
    public:
        // Packing related constants
        static constexpr ::uint8_t PACKING_FACTOR{32};
        static constexpr ::uint8_t DNA_BASE_BIT_SIZE{2};
        static constexpr ::uint64_t DNA_BASE_BITMASK{0x03};

        // Create packed sequence
        __host__ static PackedDNASequence* create();
        // Create packed sequence from file
        __host__ static PackedDNASequence* createFromFile(const ::std::string& fileName);
        // Create packed sequence from opened file
        __host__ static PackedDNASequence* createFromFile(const ::std::string& fileName, ::std::ifstream& file, ::std::optional<::size_t> numBases = ::std::nullopt, ::std::optional<::uint64_t*> bases = ::std::nullopt);

        // Move packed sequence to device
        __host__ PackedDNASequence* copyToDevice();
        // Free packed sequence
        __host__ void free() const;

        // Get number of bases
        __host__ __device__ ::size_t getNumBases() const;
        // Get number of chunks
        __host__ __device__ ::size_t getNumChunks() const;
        // Get DNA base at a given index
        __host__ __device__ DNABase operator[](::size_t idx) const;
        // Get device instance
        __host__ __device__ PackedDNASequence* getDeviceInstance() const;

        // Read packed sequence from opened file
        __host__ bool readFromFile(const ::std::string& fileName, ::std::ifstream& file);

    private:
        // Bit packed sequence implementation
        ::size_t numBases{0};
        ::size_t numChunks{0};
        ::uint64_t* bases{nullptr};
        PackedDNASequence* d_instance{nullptr};

        // Default constructor
        PackedDNASequence() = default;
        // Packed sequence constructor
        __host__ __device__ PackedDNASequence(::size_t numBases, ::size_t numChunks, ::uint64_t* bases, PackedDNASequence* d_instance = nullptr);
    };
} // cuSGA

#endif //CUSGA_PACKEDDNASEQUENCE_CUH
