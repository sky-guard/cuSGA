#ifndef CUSGA_DNABASE_CUH
#define CUSGA_DNABASE_CUH
#include <cstdint>
#include <format>
#include <stdexcept>

namespace cuSGA {
    // Define DNA base type
    using DNABase_t = ::uint8_t;

    // DNA bases related constant
    inline constexpr DNABase_t NUM_BASES{4};

    // Map from DNA base to character
    inline constexpr char DNA_BASE_TO_CHAR_MAP[]{'A', 'C', 'G', 'T'};

    // Binary representation for the 4 DNA bases to allow for bit-packing
    enum class DNABase : DNABase_t {
        A = 0b00,
        C = 0b01,
        G = 0b10,
        T = 0b11
    };

    // DNA base conversion functions
    namespace DNABaseConversion {
        // Get character for a given DNA base
        __host__ __forceinline__ char DNABaseToChar(const DNABase base) {
            auto baseValue{static_cast<DNABase_t>(base)};
            if (baseValue > NUM_BASES - 1) {
                throw ::std::runtime_error{::std::format("Unable to convert following DNA base to character: {}", baseValue)};
            }
            return DNA_BASE_TO_CHAR_MAP[baseValue];
        }

        // Get DNA base for a given character
        __host__ __forceinline__ DNABase charToDNABase(const char c) {
            switch (c) {
                case 'A': return DNABase::A;
                case 'C': return DNABase::C;
                case 'G': return DNABase::G;
                case 'T': return DNABase::T;
                default: throw ::std::runtime_error{::std::format("Unable to convert following character to DNA base: {}", c)};
            }
        }
    } // DNABaseConversion
} // cuSGA

#endif //CUSGA_DNABASE_CUH
