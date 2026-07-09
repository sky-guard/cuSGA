#include "DNABase.cuh"

#include <format>
#include <stdexcept>

namespace cuSGA {
    __host__ char DNABaseConversion::DNABaseToChar(const DNABase base) {
        // Map DNA base to character
        auto baseValue{static_cast<::uint8_t>(base)};
        if (baseValue > NUM_BASES - 1) {
            throw std::runtime_error{std::format("Unable to convert following DNA base to character: {}", baseValue)};
        }
        return DNA_BASE_TO_CHAR_MAP[baseValue];
    }

    __host__ DNABase DNABaseConversion::charToDNABase(const char c) {
        // Map character to DNA base
        switch (c) {
            case 'A': return DNABase::A;
            case 'C': return DNABase::C;
            case 'G': return DNABase::G;
            case 'T': return DNABase::T;
            default: throw std::runtime_error{std::format("Unable to convert following character to DNA base: {}", c)};
        }

    }
} // cuSGA