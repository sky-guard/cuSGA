# cuSGA: GPU-Accelerated Sequence to Graph Alignment

A **CUDA**-native implementation of the [**ParSGA**](https://github.com/ParBLiSS/ParSGA.git) Algorithm for DNA Sequence to Graph Alignment!

---

## Algorithm Implementation & Report

cuSGA is a CUDA implementation of the ParSGA Algorithm. As such, it implements five main Kernels:
* Initialization.
* Deletions.
* Substitutions.
* Insertions and Propagations.
* Minimum Cost.

While the first three and the last map quite nicely to the CUDA Parallel Architecture due to their mostly regular nature, the fourth one was for sure the most challenging to implement in a performant way and represents, for sure, the bottleneck of the program **(94% - 99% of execution time)**.

A first attempt was made by implementing a Global-level Frontier and performing a BFS-like exploration to propagate the Insertion improvements across all Nodes, disregarding Connected Components completely. While this worked, it turned out to be not performant at all, due to the excessive amount of Data Transfers between Host and Device to handle the Frontier status, as well as the Overhead Cost of launching a new Kernel for each level of the BFS. These issues were made evident by the profiling results, collected using NSight Systems and Nsight Compute, and thus demanded an architectural shift in perspective.

However, both of these problems were solved by running the BFS-like exploration algorithm in its entirety on the GPU, using only one single Kernel. Normally this wouldn't have been possible due to the exploration requiring synchronization across the entire Graph, but was in fact possible in this case exactly thanks to the presence of Connected Components.

Thus, **I decided to map one Warp per Connected Component and perform the same BFS-like exploration as before, but on a per Connected Component level**.

Here are the following reasons why:

* **Minimize thread divergence and Uncoalesced Memory Accesses**, since mapping one Thread per Connected Component could have very different results based on how uniform the Connected Components Sizes are and how far they are stored in memory.


* **Fully utilize the power of Warp-level Synchronization Primitives and Shuffling Operations**, which perfectly fit the problem of handling a Warp-level Shared Frontier.


* **Research in the papers shows that the Maximum Connected Component Size stays relatively fixed across different Pangenomes and is roughly equal to ~1500 Nodes**, which is plenty enough for a Warp to handle. **This also means that the main way the Input Size grows is not Connected Component Size, but in fact the Number of Connected Components!** So that is what the Algorithm should prioritize in its implementation.

Other optimizations were also implemented in the process, such as **Double Buffers** (used both in the Frontier and the Costs Layer in order to minimize synchronizations and memory footprint), a **Shared Memory Frontier Scratchpad** (to help boost the performance when the number of Insertions being propagated at a certain time fits below a certain fixed size, which heuristically i expect to be relatively low) and **Shared Memory Frontier Bit-Packed Queue** (which I found out could be fitted entirely in Shared Memory with a low enough memory footprint by doing some calculations).

Lastly, a final round of profiling through NSight Systems and NSight Compute showed that the situation improved significantly. Now, the remaining bottlenecks identified by the profiler are:

* Device Utilization.

  ***NOTE**: Not a real problem, read [Performance & Results](#performance--results) for more information.*

* SM Occupancy.

* Memory Access Pattern.

  ***NOTE**: This is part of the nature of the problem itself, so not much can be done to improve this. Keeping the Nodes in the Frontier sorted could potentially help with this.*

---

## Requirements & External Libraries

* **CMake** (>=4.2).
* **C++ Compiler** (with support for C++ >=23).
* **NVCC** (with support for CUDA >=23).
* [**CLI11**](https://github.com/CLIUtils/CLI11): A command line parser for C++11. 

***NOTE**: External Libraries are already handled by the CMake Build System itself. No additional libraries need to be installed manually.*

---

## Build & Usage

### 1. Clone the Repository

Clone the project's repository:
```bash
git clone https://github.com/sky-guard/cuSGA.git
```

### 2. Build with CMake

First, navigate to the project's directory (or wherever you cloned the project):
```bash
cd cuSGA
```

Set the Build Type and build using CMake:
```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

***NOTE**: Replace `-DCMAKE_BUILD_TYPE=Release` with `-DCMAKE_BUILD_TYPE=Debug` for additional debugging information.*

### 3. Run using the CLI

First, navigate to the build directory (or where your executable is located):
```bash
cd build
```

To align a sequence and print the resulting Alignment Scores, please run the following command:
```bash
cuSGA -s SEQUENCE_FILE -p PANGENOME_GRAPH_FILE -c CONNECTED_COMPONENTS_FILE_PREFIX
```

***NOTE**: For additional information regarding the expected files and their formats, please read [Input File Formats](#input-file-formats).*

For additional information, please run:
```bash
cuSGA --help
```

---

## Input File Formats

cuSGA expects the following Input Files:

* **Sequence File**: this file contains the Input Sequence(s) to be aligned to the Pangenome Graph. 

  ```
  NUMBER_OF_SEQUENCES_TO_BE_ALIGNED
  MAX_SEQUENCE_LENGTH

  SEQUENCE_LENGTH
  SEQUENCE_VALUES
  ...
  ```

  ***NOTE**: Alignment of multiple Sequences in the same file is supported, although they will be processed sequentially.*


* **Pangenome Graph File**: this file contains the Pangenome Graph to be used for the Sequence Alignments, stored in **CSR** format.

  ```
  NUMBER_OF_NODES
  NUMBER_OF_EDGES
  NODE_VALUES
  ROW_OFFSETS
  COLUMN_VALUES
  ```

* **Connected Components Files**: these files (one per DNA base) contain the information relative to the Connected Components of the Character Graphs.

  ```
  NUMBER_OF_CONNECTED_COMPONENTS
  CONNECTED_COMPONENT_OFFSETS
  NODE_MAPPINGS
  ```

  ***NOTE**: cuSGA will automatically look for these files under the following naming convention:*

  ```
  CONNECTED_COMPONENTS_FILE_NAMES_PREFIX-components-{A,C,G,T}
  ```

---

## Performance & Results

The following results were obtained testing the alignment of 100 Sequences, each one of length ~10k, to the Pangenome Graph of Salmonella, which consists roughly of ~10k nodes / edges, using a variety of Connected Component sizes:

| Implementation         | Platform                            | Execution Time | Speedup         | Time Reduction |
|:-----------------------|:------------------------------------|:---------------|:----------------|:---------------|
| **ParSGA (CPU)**       | OpenMP (8 cores)                    | 98s            | 1.0x            | 0%             |
| **CUDA-ParSGA (Ours)** | NVIDIA GPU (NVIDIA RTX 3080 Mobile) | **63s - 14s**  | **1.6x - 7.0x** | **36% - 86%**  |

**These results should however be taken with a grain of salt and are not entirely reflective of what the current implementation could be able to achieve!** 

In fact, there were a few critical problems that occurred during testing:

* **The input size is nowhere near large enough to achieve full SM occupancy**. Using some maths and profiling, I was able to deduce that (for my hardware) in order to schedule at least one block per SM, I would require an input size that is orders of magnitude larger than what I have currently available to me:

  ```
  REALISTIC_CC_SIZE(~=1000) * NUMBER_OF_WARPS_PER_BLOCK(=20) * NUMBER_OF_SM(=48) ~= 1Mil Nodes
  ```
  
  However, this is also a good thing, because it means that the **real speedup could be up to x20 - x70!** This is obviously very optimistic, but further testing needs to be done.


* **The Connected Components of varying size were simulated by grouping together smaller Components, meaning that the components generated were composed roughly of the same number of nodes. However, this is not guaranteed to be the case with real data!** Further testing using real data should be done, since this could limit the theoretical peak performance that could be achieved due to some Warps taking much longer to complete their work compared to others. 

---

## Citations & References

* **Theoretical Foundation on Sequence To Graph Alignment:**

  Jain, C., Zhang, H., Gao, Y., Aluru, S., "[*On the Complexity of Sequence to Graph Alignment*](https://doi.org/10.1007/978-3-030-17083-7_6)", 2019 Research in Computational Molecular Biology.


* **Baseline OpenMP Implementation of the ParSGA Algorithm:**

  A. Banerjee, D. Gibney, H. Xu and S. Aluru, "[*A Work-Optimal Parallel Algorithm for Aligning Sequences to Genome Graphs*](https://doi.org/10.1109/IPDPS64566.2025.00048)", 2025 IEEE International Parallel and Distributed Processing Symposium (IPDPS).