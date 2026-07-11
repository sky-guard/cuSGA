#ifndef CUSGA_APPLICATIONSETTINGS_CUH
#define CUSGA_APPLICATIONSETTINGS_CUH
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

    // Parsed arguments struct
    struct ParsedArguments {
        ::std::string sequenceFileName{};
        ::std::string pangenomeGraphFileName{};
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

        // Parse application arguments
        try {
            app.parse(argc, argv);
        } catch (const ::CLI::ParseError& parseError) {
            app.exit(parseError);
            throw;
        }

        return parsedArguments;
    }
} // cuSGA::ApplicationSettings

#endif //CUSGA_APPLICATIONSETTINGS_CUH
