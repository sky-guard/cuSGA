#ifndef CUSGA_APPLICATIONSETTINGS_CUH
#define CUSGA_APPLICATIONSETTINGS_CUH
#include <CLI/CLI.hpp>
#include <string>

#include "DNABase.cuh"

namespace cuSGA::ApplicationSettings {
    // Application name
    inline constexpr auto APP_NAME{"cuSGA"};
    inline constexpr auto APP_DESCRIPTION{"A CUDA-native implementation of the ParSGA Algorithm for DNA Sequence to Graph Alignment!"};

    // Application version
    inline constexpr auto VERSION_FLAG{"-v,--version"};
    inline constexpr auto APP_VERSION{"1.0"};

    // Sequence file name
    inline constexpr auto SEQUENCE_FILE_NAME_FLAG{"-s,--sequence"};
    inline constexpr auto SEQUENCE_FILE_NAME_DESCRIPTION{"The path to the file containing the DNA Sequence to align."};

    // Pangenome graph file name
    inline constexpr auto PANGENOME_GRAPH_FILE_NAME_FLAG{"-p,--pangenome-graph"};
    inline constexpr auto PANGENOME_GRAPH_FILE_NAME_DESCRIPTION{"The path to the file containing the CSR representation of the Pangenome Graph to align to."};

    // Connected components file names
    inline constexpr auto CONNECTED_COMPONENTS_FILE_NAMES_FLAG{"-c,--connected-components"};
    inline constexpr auto CONNECTED_COMPONENTS_FILE_NAMES_DESCRIPTION{"The paths to the files containing the Connected Components of the Character Graphs for the given Pangenome Graph. The common prefix of the file names is also sufficient and cuSGA will automatically look for files with the following names: $PREFIX-components-{A, C, G, T}."};
    inline constexpr ::std::string CONNECTED_COMPONENTS_FILE_NAME_SUFFIXES[]{"-components-A", "-components-C", "-components-G", "-components-T"};

    // Parsed arguments struct
    struct ParsedArguments {
        ::std::string sequenceFileName{};
        ::std::string pangenomeGraphFileName{};
        ::std::string connectedComponentsFileNames[NUM_BASES]{};
    };

    inline ParsedArguments parseArguments(const int argc, const char* const argv[]) {
        // Create application instance
        ::CLI::App app{APP_DESCRIPTION, APP_NAME};

        // Configure application version
        app.set_version_flag(VERSION_FLAG, APP_VERSION);

        // Create parsed application arguments container
        ParsedArguments parsedArguments{};

        // Temporary container for character graph file names
        ::std::vector<::std::string> connectedComponentsFileNamesVector{};

        // Configure application arguments
        app.add_option(SEQUENCE_FILE_NAME_FLAG, parsedArguments.sequenceFileName, SEQUENCE_FILE_NAME_DESCRIPTION)->required();
        app.add_option(PANGENOME_GRAPH_FILE_NAME_FLAG, parsedArguments.pangenomeGraphFileName, PANGENOME_GRAPH_FILE_NAME_DESCRIPTION)->required();
        const auto connectedComponentsFileNamesOption{app.add_option(CONNECTED_COMPONENTS_FILE_NAMES_FLAG, connectedComponentsFileNamesVector, CONNECTED_COMPONENTS_FILE_NAMES_DESCRIPTION)->expected(1, NUM_BASES)};

        // Parse application arguments
        try {
            app.parse(argc, argv);
        }
        catch (const ::CLI::ParseError& parseError) {
            app.exit(parseError);
            throw;
        }

        // Deduce connected components file names from suffix if only one path is given as input
        if (connectedComponentsFileNamesOption->count() == 1) {
            // Get prefix
            const auto prefix{std::move(connectedComponentsFileNamesVector[0])};

            // Deduce file names
            connectedComponentsFileNamesVector.resize(NUM_BASES);
#pragma unroll
            for (DNABase_t i = 0; i < NUM_BASES; ++i) {
                connectedComponentsFileNamesVector[i] = prefix + CONNECTED_COMPONENTS_FILE_NAME_SUFFIXES[i];
            }
        }
        // Throw exception if not all connected components file names have been given as input
        else if (connectedComponentsFileNamesOption->count() != 4) {
            throw ::std::runtime_error{"The number of Connected Components files provided does not match the number of DNA Bases. Please provide one Connected Components file for each base in the following order: A, C, G, T. Alternatively, let cuSGA deduce the file names by only providing the common prefix."};
        }

        // Move over connected components file names data
#pragma unroll
        for (DNABase_t baseIdx{0}; baseIdx < NUM_BASES; ++baseIdx) {
            parsedArguments.connectedComponentsFileNames[baseIdx] = ::std::move(connectedComponentsFileNamesVector[baseIdx]);
        }

        return parsedArguments;
    }
} // cuSGA::ApplicationSettings

#endif //CUSGA_APPLICATIONSETTINGS_CUH
