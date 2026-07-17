#include <format>

#include "ApplicationSettings.cuh"
#include "SequenceGraph.cuh"

int main(const int argc, const char* const argv[]) {
    try {
        // Parse arguments
        ::cuSGA::ApplicationSettings::ParsedArguments parsedArguments{};
        try {
            parsedArguments = ::cuSGA::ApplicationSettings::parseArguments(argc, argv);
        } catch (const ::CLI::ParseError& parseError) {
            return parseError.get_exit_code();
        }

        // Create sequence graph instance
        const auto sequenceGraph{::cuSGA::SequenceGraph::createFromFiles(parsedArguments.pangenomeGraphFileName, parsedArguments.connectedComponentsFileNames)};

        // Align sequence to graph
        const auto scores{sequenceGraph->align(parsedArguments.sequenceFileName)};

        // Print results
        ::std::cout << "cuSGA Alignment Scores: " << std::endl;
        for (const auto score: scores) {
            ::std::cout << ::std::format("{} ", score);
        }

        // Free memory
        sequenceGraph->free();

        return 0;
    } catch (const ::std::exception& exception) {
        // Handle exceptions
        ::std::cerr << exception.what() << ::std::endl;

        return EXIT_FAILURE;
    }
}
