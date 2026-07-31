# cuSGA: GPU-Accelerated Sequence to Graph Alignment

A **CUDA**-native implementation **(and revision)** of the [**ParSGA**](https://github.com/ParBLiSS/ParSGA.git) Algorithm for DNA Sequence to Graph Alignment!

---

## Algorithm Implementation & Report

cuSGA is a CUDA implementation (and revision) of the ParSGA Algorithm. As such, it implements five main Kernels:
* Initialization.
* Deletions.
* Substitutions.
* **Insertions and Propagations.**
* Minimum Cost.

The first three kernels and the last one map quite nicely to the CUDA Parallel Architecture due to their mostly regular nature, making them quite straightforward and easy to implement. **The fourth kernel, on the other hand, deserves some more attention**, as it was for sure the most challenging one to implement in a performant way and represents the bottleneck of the entire program **(94% - 99% of execution time)**.

A first parallelization attempt was made by implementing a **Global-level Frontier and performing a BFS-like exploration over the entire Graph** to propagate the Insertion improvements across all Nodes, disregarding Connected Components completely. While this worked, it turned out to be **not performant at all**, due to the excessive amount of Data Transfers between Host and Device necessary to handle the Frontier status, as well as the Overhead Cost of launching a new Kernel instance for each level of the BFS. These issues were made evident by the profiling results, collected using NSight Systems and Nsight Compute, and thus demanded an architectural shift in perspective.

Both of these problems stemmed from the fact that **performing an exploration over the entirety of the Graph all at once inherently required some form of Grid-level synchronization, which proved to be too costly to achieve in this case**. **However, thanks to the presence of Connected Components, Grid-level synchronization was in fact not necessary at all: the exploration could instead be carried out across the Nodes of each Connected Component individually, in a completely independent manner**. This turned out to be key to solve this challenge, as it made possible to run the BFS-like exploration algorithm in its entirety on the GPU all at once, using only a single Kernel launch per layer.

Thus, **I made the decision to map one Warp per Connected Component and perform the same BFS-like exploration as before, but at a per Connected Component level**. This is because **mapping one Thread per Connected Component would have been too costly** for a few different reasons:

* **Inability to exploit parallelism inside the exploration of the single Connected Component**, whose size, according to the research, could be up to 1500 - 2000 Nodes, which is plenty for parallelism to be exploited.

* **Thread Divergence and Uncoalesced Memory Accesses across Threads of the same Warp**, due to varying Connected Component sizes and how Connected Components data is stored in memory and meant to be accessed. 

* **Inability to fully utilize the power of Warp-level Synchronization Primitives and Shuffling Operations**, which can be used to efficiently coordinate exploration using a shared Warp-level Frontier.

On the other hand, **mapping one Block per Connected Component would have been equally inefficient**:

* **Inability to exploit parallelism along the real dimension of the problem**, which is not Connected Component Size (which stays relatively fixed, as shown by research in the papers), but rather the number of Connected Components that can be processed in parallel at the same time.

It would take me far too long to list all the possible optimizations and areas of improvement that were taken into consideration during the development of this project, however here are some of the most important to note and that had a positive effective in helping to boost the final performance of the program:

* **Bit-Packing**, in order to reduce the memory footprint and transfer times of linear data structures.

* **Double-Buffering**, used both in the Frontier and Costs layers to minimize synchronizations and memory footprint.

* **Usage of Shared Memory**, utilized wherever possible in order to reduce Global Memory access latency, while also keeping track of the overall memory footprint of what was stored inside it, in order for it not to be a limiting factor when determining final SM Occupancy.

* **Run-Time Block Size Fine-Tuning**, to dynamically size Blocks based on the current hardware capabilities through the provided CUDA Occupancy API.

* **Register Occupancy Fine-Tuning**, to minimize Register Usage, which turned out to be one of the limiting factors for the kernel's performance, and make sure no Local Memory Spills were happening. 

* **Minimization of Synchronization points and Usage of Asynchronous Operations through Pinned Memory**, further improved by using a custom Linearized Data Layout on the Host Side to speed up Data Transfers and minimize Driver Overhead as much as possible.

* **Loop Unrolling and Function Inlining**, to help squeeze as much performance as possible from the compiler, keeping an eye on Register Occupancy in order to avoid Local Memory Spilling.

Lastly, a final round of profiling through NSight Systems and NSight Compute showed that the situation improved significantly. **The remaining bottlenecks identified by the profiler are the following**:

* **Device Utilization**.

  ***NOTE**: Not a real problem, as this is caused by insufficient input size. Read [Performance & Results](#performance--results) for more information.*

* **SM Occupancy**.

  ***NOTE**: Further testing needs to be done to determine whether this is a real problem or not. NCU reported the actual number of Active Threads being much lower than the expected theoretical limit of around 75% (caused by a somewhat high Register Usage of 58 Registers per Thread). However, this could partially be caused by the insufficient input size, as stated in [Performance & Results](#performance--results), as well as the number of Insertions that can be propagated at the same time being lower than the size of a Warp. This is, heuristically, to be expected, but could be mitigated by larger Connected Component Sizes.*

* **Memory Access Pattern**.

  ***NOTE**: As this is part of the nature of the problem itself, not much can be done to improve this. Keeping the Nodes in the Frontier sorted could potentially help.*

One final parallelization attempt was made, this time **going back to the original idea of a Grid level Synchronous exploration**. In fact, the original implementation turned out to be quite lucklaster not exactly because it was a bad idea, but in fact becasue it could have been implemented way better: instead of launching one Kernel per level of the exploration and transferring Frontier data back and forth from Device to Host, **the same idea could have been implemented using Cooperative Groups, thus resolving both of the aformentioned issues**. 

This idea was also sparked by the following realization about the original ParSGA Algorithm: **it is possible to determine at Substitutions time exactly which Nodes / Connected Components will need to be visited in the following Insertions phase**. In fact, **a Node needs to be visited for Insertions / Propagations only if there is an improvement at Substitutions time!** This information can then be used in order to avoid a full scan of the Graph at the start of the Insertions phase to find the initial Frontier.

The same idea also obviously can be applied to the Connected Components level Synchronization version of the Algorithm, although with diminishing returns as opposed to the Grid level Synchronization version, which in fact turned out to be the **most successful parallelization attempt, as well as the most consistent, since it doesn't rely on Connected Components at all.** 

Thus, **I propose this revised version of the Algorithm as the one to be used (by default) by cuSGA**. **Could this be a definitive improvement over the original ParSGA Algorithm, even in a CPU-only runtime scenario?** More testing would have to be done against an OpenMP implementation of the same Algorithm to find out the answer.

Finally, let's review the results and bottlenecks identified by NSight Systems and NSight Compute:

* **Excessive Stalling and Low Active Threads Occupancy**.

  ***NOTE**: This isn't surprising at all and is, in fact, to be expected. Thanks to the aforementioned optimizations, it was possible to efficiently reduce the size of Global Queue to a minimum. This, combined with the fact that we are now using Cooperative Groups to fill SMs as much as possible, means that a very large number of Threads will just either overflow the Global Queue Size or be inactive due to a too small number of Insertions that can be propagated in parallel at any given time. While the first issue should be mitigated by increasing (significantly) the size of the input, the second one is much harder to solve, as it is part of the nature of the problem itself. Another secondary factor that could contribute to this (though not in this case, as the input was quite regular in this regard), is the Variance of the Neighbor Out-Degree. This is because each Thread is mapped to a single Node, and Nodes can have a varying number of Neighbors.*

* **Inability to reach Theoretical Occupancy and Work Imbalance**. 

  ***NOTE**: These issues, although present, are very much minor compared to the first one. This is coherent with the results shown by NCU (around ~20% estimated Speedup VS ~75% estimated Speedup for Stalls). Although Maximum Occupancy in this case is only 75% due to the high Register Usage, the final number was overall still quite good (9 Warps per Scheduler VS 12 Theoretical Maximum). The same can be also said about Work Imbalance (Maximum 45% above average, Minimum 18% below average).*

Some final conclusions and thoughts: **if more testing were to be done, analyzing how many Insertions can be propagated on average in parallel across all the Nodes, and if this number were big enough... then parallelizing this Algorithm on a GPU could be considered worth it. But if instead, this number were quite low... then that would mean that this Problem is mostly Serial in nature and thus attempting to parallelize it on a massively-parallel throughput-focused device is bound to be mostly unsuccessful**. 

***NOTE**: Some quick (and informal) testing seems to suggest that the number of Insertions that can be propagated in parallel consists of only around 2-3% of the overall Graph Size. Assuming that the input Graph Size were able to scale to a big enough number, which should indeed be the case with complex genomes, this could make the parallelization worth it.*

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

### 2. Build using CMake

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

To align a sequence using **Grid level Synchronization with Cooperative Groups (DEFAULT)** and print the resulting Alignment Scores, please run the following command:
```bash
./cuSGA -s SEQUENCE_FILE -p PANGENOME_GRAPH_FILE -c CONNECTED_COMPONENTS_FILE_PREFIX
```

***NOTE**: **For very large Graphs, it is recommended to pass the ```--block-aggregation``` flag** to enable a variant of the Grid level Synchronization Kernel that additionally also makes use of Block Aggregation (on top of the already existing Warp Aggregation). This is to further improve Scalability by focusing on minimizing the concurrent atomic writes on the Global Queue Size, which is expected to become more and more of a limiting factor for performance as the input size grows. **For smaller graphs, such as the one used during testing, performance loss appears to be minimal (10.8x speedup vs 11.2x speedup).***

To align a sequence using **Connected Components level Synchronization (NOT RECOMMENDED)** and print the resulting Alignment Scores, please run the following command:
```bash
./cuSGA --cc-align -s SEQUENCE_FILE -p PANGENOME_GRAPH_FILE -c CONNECTED_COMPONENTS_FILE_PREFIX
```

***NOTE**: For additional information regarding the expected files and their formats, please read [Input File Formats](#input-file-formats).*

For additional information, please run:
```bash
./cuSGA --help
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
  CONNECTED_COMPONENT_REVERSE_MAPPINGS
  ```

  ***NOTE**: cuSGA will automatically look for these files under the following naming convention:*

  ```
  CONNECTED_COMPONENTS_FILE_NAMES_PREFIX-components-{A,C,G,T}
  ```

---

## Performance & Results

The following results were obtained testing the alignment of 100 Sequences, each one of length ~10k, to the Pangenome Graph of Salmonella, which consists roughly of ~10k nodes / edges, using a variety of Connected Component sizes:

| Implementation | Platform                            | Execution Time    | Speedup          | Time Reduction  |
|:---------------|:------------------------------------|:------------------|:-----------------|:----------------|
| **ParSGA**     | OpenMP (Ryzen 9 6900HS, 8 cores)    | 95s               | 1.0x             | 0%              |
| **cuSGA**      | CUDA (NVIDIA RTX 3080 Mobile)       | **120.2s - 8.5s** | **0.8x - 11.2x** | **+26% - -91%** |

**These results should however be taken with a grain of salt and are not entirely reflective of what the current implementation could be able to achieve!** 

In fact, there were a few critical problems that occurred during testing:

* **The current input size was nowhere near large enough to achieve full SM occupancy**. The **0.8x speedup** was achieved under the very unfavorable case of **Connected Components level Synchronization** with scheduling **only a single block**, while the **11.2x speedup** was achieved by using **Grid level Synchronization**, which proved to be **more consistent (since it doesn't make use of Connected Components at all)** but still **Active Thread Occupancy proved to be extremely poor due to the minuscule amount of Insertions that needed propagation**. Therefore, if the input size were big enough to saturate all SMs, **cuSGA could theoretically achieve numbers that are much higher than this**, assuming Memory Bandwidth doesn't become the limiting factor. Using some maths and profiling, it was possible for me to deduce that in order to schedule at least one block per SM (according to my hardware), the input size would have to be orders of magnitude larger than the one that was used in the current testing:

  ```
  REALISTIC_CC_SIZE(~=1000) * NUMBER_OF_WARPS_PER_BLOCK(=48) * NUMBER_OF_SM(=48) ~= 750k Nodes
  ```

* **The Connected Components of varying size were simulated by grouping together smaller Components, meaning that the components generated were composed roughly of the same number of nodes**. **However, this is not guaranteed to be the case with real data!** Further testing using actual data should be done, as this could be a **limiting factor for the theoretical peak performance of the Connected Components level Synchronization Algorithm**: if the Connected Component Sizes were highly irregular, some Warps could end up taking a lot longer to complete their work compared to other Warps in the same Block, hogging hardware resources until the whole Block finishes its computation. **The Grid level Synchronization Algorithm on the other hand, would remain completely unaffected, thus possibly offering better scalability, especially in the case of big Graphs!.**

---

## Citations & References

* **Theoretical Foundation on Sequence To Graph Alignment:**

  Jain, C., Zhang, H., Gao, Y., Aluru, S., "[*On the Complexity of Sequence to Graph Alignment*](https://doi.org/10.1007/978-3-030-17083-7_6)", 2019 Research in Computational Molecular Biology.


* **Baseline OpenMP Implementation of the ParSGA Algorithm:**

  A. Banerjee, D. Gibney, H. Xu and S. Aluru, "[*A Work-Optimal Parallel Algorithm for Aligning Sequences to Genome Graphs*](https://doi.org/10.1109/IPDPS64566.2025.00048)", 2025 IEEE International Parallel and Distributed Processing Symposium (IPDPS).