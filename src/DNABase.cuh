#ifndef CUSGA_DNABASE_CUH
#define CUSGA_DNABASE_CUH
#include <cstdint>

namespace cuSGA {
    // DNA bases related constant
    inline constexpr ::size_t NUM_BASES{4};

    // Map from DNA base to character
    inline constexpr char DNA_BASE_TO_CHAR_MAP[]{'A', 'C', 'G', 'T'};

    // Binary representation for the 4 DNA bases to allow for bit-packing
    enum class DNABase : ::uint8_t {
        A = 0b00,
        C = 0b01,
        G = 0b10,
        T = 0b11
    };

    namespace DNABaseConversion {
        // Get character for a given DNA base
        __host__ char DNABaseToChar(DNABase base);
        // Get DNA base for a given character
        __host__ DNABase charToDNABase(char c);
    } // DNABaseConversion
} // cuSGA

#endif //CUSGA_DNABASE_CUH
