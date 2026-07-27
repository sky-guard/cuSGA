cmake_minimum_required(VERSION 4.2)
project(cuSGA CXX CUDA)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CUDA_STANDARD 23)

set(CMAKE_CUDA_ARCHITECTURES native)

include(FetchContent)

FetchContent_Declare(
        cli11
        GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git
        GIT_TAG v2.6.2
)

FetchContent_MakeAvailable(cli11)

add_executable(cuSGA src/main.cu
        src/PangenomeGraph.cu
        src/PangenomeGraph.cuh
        src/DNABase.cuh
        src/PackedDNASequence.cu
        src/PackedDNASequence.cuh
        src/SequenceGraph.cu
        src/SequenceGraph.cuh
        src/DoubleBuffer.cuh
        src/ApplicationSettings.cuh
        src/KernelUtils.cuh
        src/Frontier.cuh
        src/Utils.cuh
)

set_target_properties(cuSGA PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_compile_options(cuSGA PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
target_compile_options(cuSGA PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--ptxas-options=-v>)

target_link_libraries(cuSGA PRIVATE CLI11::CLI11)
