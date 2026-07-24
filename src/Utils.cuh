#ifndef CUSGA_UTILS_CUH
#define CUSGA_UTILS_CUH
#include <format>
#include <fstream>
#include <string>

namespace cuSGA {
    // Define byte type
    using byte_t = unsigned char;

    // Define target size type
    using targetSize_t = ::uint32_t;

    namespace Utils {
        // Utils related constants
        inline constexpr ::uint8_t BYTE_SIZE{8};

        // Open file and check for errors
        __host__ __forceinline__ ::std::ifstream openFile(const std::string& fileName) {
            // Create input file stream
            ::std::ifstream file{fileName};

            // Open file
            if (!file.is_open()) {
                throw ::std::runtime_error{::std::format("Unable to open file: {}", fileName)};
            }

            return file;
        }
    } // cuSGA::Utils

} // cuSGA

#endif //CUSGA_UTILS_CUH
