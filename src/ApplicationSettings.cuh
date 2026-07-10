#ifndef CUSGA_APPLICATIONSETTINGS_CUH
#define CUSGA_APPLICATIONSETTINGS_CUH
#include "DNABase.cuh"

#include <CLI/CLI.hpp>
#include <string>

namespace cuSGA::ApplicationSettings {
    // Application name
    constexpr auto APP_NAME{"cuSGA"};
    constexpr auto APP_DESCRIPTION{"A CUDA-native implementation of the ParSGA Algorithm for DNA Sequence to Graph Alignment!"};

    // Application version
    constexpr auto VERSION_FLAG{"-v,--version"};
    constexpr auto APP_VERSION{"1.0"};

    // Sequence file name
    constexpr auto SEQUENCE_FILE_NAME_FLAG{"-s,--sequence"};
    constexpr auto SEQUENCE_FILE_NAME_DESCRIPTION{"The path to the file containing the DNA Sequence to align."};

    // Pangenome graph file name
    constexpr auto PANGENOME_GRAPH_FILE_NAME_FLAG{"-p,--pangenome-graph"};
    constexpr auto PANGENOME_GRAPH_FILE_NAME_DESCRIPTION{"The path to the file containing the CSR representation of the Pangenome Graph to align to."};

    // Character graph file names
    constexpr auto CHARACTER_GRAPH_FILE_NAMES_FLAG{"-c,--character-graphs"};
    constexpr auto CHARACTER_GRAPH_FILE_NAMES_DESCRIPTION{"The paths to the files containing the (partial) CSR representations of the Character Graphs for the given Pangenome Graph. If not specified, cuSGA will compute them starting from the given Pangenome Graph. Additionally, it's possible to specify only the common prefix of the file names and cuSGA will automatically look for files with the following names: $PREFIX_{A, C, G, T}.csr."};
    constexpr ::std::string CHARACTER_GRAPH_FILE_NAME_SUFFIXES[]{"_A.csr", "_C.csr", "_G.csr", "_T.csr"};

    // Parsed arguments struct
    struct ParsedArguments {
        ::std::string sequenceFileName{};
        ::std::string pangenomeGraphFileName{};
        ::std::string characterGraphFileNames[NUM_BASES]{};
        bool computeCharacterGraphs = false;
    };

    inline ParsedArguments parseArguments(const int argc, const char* const argv[]) {
        // Create application instance
        ::CLI::App app{APP_DESCRIPTION, APP_NAME};

        // Configure application version
        app.set_version_flag(VERSION_FLAG, APP_VERSION);

        // Create parsed application arguments container
        ParsedArguments parsedArguments{};

        // Temporary container for character graph file names
        ::std::vector<::std::string> characterGraphFileNamesVector{};

        // Configure application arguments
        app.add_option(SEQUENCE_FILE_NAME_FLAG, parsedArguments.sequenceFileName, SEQUENCE_FILE_NAME_DESCRIPTION)->required();
        app.add_option(PANGENOME_GRAPH_FILE_NAME_FLAG, parsedArguments.pangenomeGraphFileName, PANGENOME_GRAPH_FILE_NAME_DESCRIPTION)->required();
        const auto characterGraphFileNamesOption{app.add_option(CHARACTER_GRAPH_FILE_NAMES_FLAG, characterGraphFileNamesVector, CHARACTER_GRAPH_FILE_NAMES_DESCRIPTION)->expected(0, NUM_BASES)};

        // Parse application arguments
        try {
            app.parse(argc, argv);
        } catch (const ::CLI::ParseError& parseError) {
            app.exit(parseError);
            throw;
        }

        // Set flag to compute and store character graphs if no path is given as input
        if (characterGraphFileNamesOption->count() == 0) {
            // Set compute character graphs flag
            parsedArguments.computeCharacterGraphs = true;

            // Set pangenome graph file name as default prefix for character graph file names
            characterGraphFileNamesVector.push_back(parsedArguments.pangenomeGraphFileName);
        }
        // Deduce character graph file names from suffix if only one path is given as input
        if (characterGraphFileNamesOption->count() <= 1) {
            // Get prefix
            const auto prefix{characterGraphFileNamesVector[0]};

            // Deduce file names
            characterGraphFileNamesVector.resize(NUM_BASES);
            for (::size_t i = 0; i < NUM_BASES; ++i) {
                characterGraphFileNamesVector[i] = prefix + CHARACTER_GRAPH_FILE_NAME_SUFFIXES[i];
            }
        }
        // Throw exception if not all character graph file names have been given as input
        else if (characterGraphFileNamesOption->count() != 4) {
            throw ::std::runtime_error{"The number of Character Graph files provided does not match the number of DNA Bases. Please provide one Character Graph file for each base in the following order: A, C, G, T. Alternatively, let cuSGA deduce the file names by only providing the common prefix or let cuSGA compute them from scratch for you by omitting this flag."};
        }

        // Move over character graph file names data
        for (::size_t i{0}; i < NUM_BASES; ++i) {
            parsedArguments.characterGraphFileNames[i] = ::std::move(characterGraphFileNamesVector[i]);
        }

        return parsedArguments;
    }
} // cuSGA::ApplicationSettings

#endif //CUSGA_APPLICATIONSETTINGS_CUH
