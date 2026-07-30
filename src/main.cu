#include <format>

#include "ApplicationSettings.cuh"
#include "SequenceGraph.cuh"

int main(const int argc, const char* const argv[]) {
    try {
        // Parse arguments
        ::cuSGA::ApplicationSettings::ParsedArguments parsedArguments{};
        try {
            parsedArguments = ::cuSGA::ApplicationSettings::parseArguments(argc, argv);
        }
        catch (const ::CLI::ParseError& parseError) {
            return parseError.get_exit_code();
        }

        // Create sequence graph instance
        auto sequenceGraph{::cuSGA::SequenceGraph{parsedArguments.pangenomeGraphFileName, parsedArguments.sequenceFileName, parsedArguments.connectedComponentsFileNames, true}};

        // Align sequence to graph
        const auto scores{(parsedArguments.useGridAlignment)? sequenceGraph.gridAlign(parsedArguments.sequenceFileName) : sequenceGraph.connectedComponentsAlign(parsedArguments.sequenceFileName)};

        // Print results
        ::std::cout << "cuSGA Alignment Scores: " << std::endl;
        for (::cuSGA::scoreSize_t scoreIdx = 0; scoreIdx < sequenceGraph.getNumScores(); ++scoreIdx) {
            ::std::cout << ::std::format("{} ", scores[scoreIdx]);
        }

        // Free memory
        sequenceGraph.free();

        return 0;
    }
    catch (const ::std::exception& exception) {
        // Handle exceptions
        ::std::cerr << exception.what() << ::std::endl;

        return EXIT_FAILURE;
    }
}
