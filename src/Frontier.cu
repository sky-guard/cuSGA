#include "Frontier.cuh"

namespace cuSGA {
    __host__ Frontier::Frontier(const ::size_t size, const bool ownsInstance, Frontier* const pinned_instanceOptional, KernelUtils::BumpPtrAllocator* const allocatorOptional) : Frontier(0, 0, DoubleBuffer<::size_t>{}, nullptr, ownsInstance, pinned_instanceOptional) {
        // Get allocator
        KernelUtils::BumpPtrAllocator* allocator{allocatorOptional};
        KernelUtils::BumpPtrAllocator allocatorInstance{};
        if (!allocator) {
            allocator = &allocatorInstance;
        }

        // Grow allocator
        if (ownsInstance) {
            allocator->grow<Frontier>();
            growBuffers(allocator, size);
        }

        // Initialize allocator
        if (ownsInstance) {
            allocator->initHostPinnedMem();
        }

        // Emplace buffers
        if (ownsInstance) {
            this->pinned_instance = allocator->emplaceReserve<Frontier>();
        }
        this->doubleBuffer = DoubleBuffer{size, false, &pinned_instance->doubleBuffer, allocator};
        this->isInQueue = allocator->emplaceSet<::std::remove_pointer_t<decltype(isInQueue)>>(0, size);
    }

    __host__ Frontier Frontier::copyToDevice(Frontier* const d_instanceOptional, KernelUtils::BumpPtrAllocator* allocatorOptional) {
        // Check if device instance already exists for this frontier
        if (d_instance) {
            throw ::std::runtime_error{"Device instance already exists for this Frontier!"};
        }

        // Get allocator
        KernelUtils::BumpPtrAllocator* allocator{allocatorOptional};
        KernelUtils::BumpPtrAllocator allocatorInstance{};
        if (!allocator) {
            allocator = &allocatorInstance;
        }

        // Grow allocator
        if (ownsInstance) {
            allocator->grow<Frontier>();
            growBuffers(allocator, doubleBuffer.getSize());
        }

        // Initialize allocator
        if (ownsInstance) {
            allocator->initCudaGMem();
        }

        // Reserve instance
        if (ownsInstance) {
            this->d_instance = allocator->emplaceReserve<Frontier>();
        }
        else {
            this->d_instance = d_instanceOptional;
        }

        // Emplace buffers
        const auto d_doubleBuffer{doubleBuffer.copyToDevice(&d_instance->doubleBuffer, allocator)};
        const auto d_isInQueue{allocator->cudaEmplaceCopy<::std::remove_pointer_t<decltype(isInQueue)>>(isInQueue, ::cudaMemcpyHostToDevice, doubleBuffer.getSize(), false, cudaStreamDefault)};

        // Create temporary host instance holding the device pointers
        const Frontier d_frontier{currentSize, alternateSize, d_doubleBuffer, d_isInQueue, ownsInstance, pinned_instance,d_instance};

        // Emplace instance
        if (ownsInstance) {
            *pinned_instance = d_frontier;
            CUDA_CHECK(::cudaMemcpyAsync(d_instance, pinned_instance, sizeof(Frontier), ::cudaMemcpyHostToDevice, cudaStreamDefault));
        }

        return d_frontier;
    }

    __host__ void Frontier::free() const {
        // Free device memory if present
        if (d_instance) {
            if (ownsInstance) {
                CUDA_CHECK(::cudaFreeAsync(d_instance, cudaStreamDefault));
            }
            else {
                CUDA_CHECK(::cudaFreeAsync(pinned_instance->getBuffersRoot(), cudaStreamDefault));
            }
        }

        // Free host memory
        if (ownsInstance) {
            CUDA_CHECK(::cudaFreeHost(pinned_instance));
        }
        else {
            CUDA_CHECK(::cudaFreeHost(getBuffersRoot()));
        }
    }
} //cuSGA