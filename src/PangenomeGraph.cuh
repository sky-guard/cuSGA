#ifndef CUSGA_PANGENOMEGRAPH_CUH
#define CUSGA_PANGENOMEGRAPH_CUH
#include <string>

#include "DNABase.cuh"
#include "PackedDNASequence.cuh"

namespace cuSGA {
    class PangenomeGraph {
    public:
        // Create pangenome graph from file
        __host__ static PangenomeGraph* createFromFile(const std::string& fileName);
        // Create pangenome graph from opened file
        __host__ static PangenomeGraph* createFromFile(const std::string& fileName, std::ifstream& file, std::optional<::size_t> numNodes = std::nullopt, std::optional<::size_t> numEdges = std::nullopt, std::optional<PackedDNASequence*> baseValues = std::nullopt, std::optional<const ::size_t*> columnValues = std::nullopt, std::optional<const ::size_t*> rowOffsets = std::nullopt);

        // Move pangenome graph to device
        __host__ PangenomeGraph* copyToDevice();
        // Free pangenome graph
        __host__ void free(bool freeSequence = false) const;

        // Get number of nodes in the graph
        __host__ __device__ ::size_t getNumNodes() const;
        // Get number of edges in the graph
        __host__ __device__ ::size_t getNumEdges() const;
        // Get DNA base values for the graph
        __host__ __device__ PackedDNASequence* getBaseValues() const;
        // Get DNA base value for a given node index
        __host__ __device__ DNABase getDNABase(::size_t nodeIdx) const;
        // Get neighbors for a given node index
        __host__ __device__ const ::size_t* getNeighbors(::size_t nodeIdx) const;
        // Get number of neighbors for a given node index
        __host__ __device__ ::size_t getNumNeighbors(::size_t nodeIdx) const;
        // Get device instance
        __host__ __device__ PangenomeGraph* getDeviceInstance() const;

    private:
        // CSR pangenome graph implementation
        ::size_t numNodes{0};
        ::size_t numEdges{0};
        PackedDNASequence* baseValues{nullptr};
        const ::size_t* columnValues{nullptr};
        const ::size_t* rowOffsets{nullptr};
        PangenomeGraph* d_instance{nullptr};

        // Default constructor
        PangenomeGraph() = default;
        // Pangenome graph constructor
        __host__ __device__ PangenomeGraph(::size_t numNodes, ::size_t numEdges, PackedDNASequence* baseValues, const ::size_t* columnValues, const ::size_t* rowOffsets, PangenomeGraph* d_instance = nullptr);
    };
} // cuSGA

#endif //CUSGA_PANGENOMEGRAPH_CUH
