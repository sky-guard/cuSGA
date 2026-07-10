#include "Frontier.cuh"

namespace cuSGA {
    __host__ Frontier* Frontier::create(const ::uint64_t size) {
        // Allocate double buffer
        const auto doubleBuffer{DoubleBuffer<::uint64_t>::create(size)};

        // Allocate isInQueue
        const auto isInQueue{new cuda::atomic<bool, cuda::thread_scope_device>[size]{false}};

        // Create frontier instance
        const auto frontier{new Frontier{0, 0, doubleBuffer, isInQueue}};

        return frontier;
    }

    __host__ Frontier* Frontier::copyToDevice() {
        // Check if device instance already exists for this frontier
        if (d_instance) {
            return d_instance;
        }

        // Allocate device frontier buffers
        cuda::atomic<bool, cuda::thread_scope_device>* d_isInQueue{nullptr};
        KernelUtils::cudaMalloc(&d_isInQueue, doubleBuffer->getSize() * sizeof(cuda::atomic<bool, cuda::thread_scope_device>));

        // Copy buffers data from host to device
        const auto d_doubleBuffer{doubleBuffer->copyToDevice()};
        KernelUtils::cudaMemcpy(d_isInQueue, isInQueue, doubleBuffer->getSize() * sizeof(cuda::atomic<bool, cuda::thread_scope_device>), ::cudaMemcpyHostToDevice);

        // Allocate device frontier instance
        Frontier* d_frontier{nullptr};
        KernelUtils::cudaMalloc(&d_frontier, sizeof(Frontier));

        // Create temporary host instance holding the device pointers
        const Frontier deviceFrontier{currentSize, alternateSize.load(cuda::memory_order_relaxed), d_doubleBuffer, d_isInQueue, d_frontier};

        // Update host instance data
        this->d_instance = d_frontier;

        // Copy instance data from host to device
        KernelUtils::cudaMemcpy(d_frontier, &deviceFrontier, sizeof(Frontier), ::cudaMemcpyHostToDevice);

        return d_frontier;
    }

    __host__ void Frontier::free() const {
        // Free device memory if device instance is present
        if (d_instance) {
            // Create a temporary host copy of the device instance to get internal pointers
            Frontier deviceFrontier{};
            KernelUtils::cudaMemcpy(&deviceFrontier, d_instance, sizeof(Frontier), ::cudaMemcpyDeviceToHost);

            // Free device frontier buffers
            if (deviceFrontier.isInQueue) {
                KernelUtils::cudaFree(deviceFrontier.isInQueue);
            }

            // Free device frontier instance
            KernelUtils::cudaFree(d_instance);
        }

        // Free host memory
        if (doubleBuffer) {
            this->doubleBuffer->free();
        }
        delete[] isInQueue;
        delete this;
    }

    __host__ __device__ ::uint64_t Frontier::getSize() const {
        return currentSize;
    }

    __host__ __device__ bool Frontier::isEmpty() const {
        return currentSize == 0;
    }

    __host__ bool Frontier::isEmptySync() {
        // Copy back current size from device
        if (d_instance) {
            KernelUtils::cudaMemcpy(&this->currentSize, &d_instance->currentSize, sizeof(currentSize), ::cudaMemcpyDeviceToHost);
        }

        return currentSize == 0;
    }

    __host__ __device__ ::uint64_t Frontier::getNodeIndex(const ::uint64_t idx) const {
        return doubleBuffer->current()[idx];
    }

    __host__ __device__ Frontier* Frontier::getDeviceInstance() const {
        return d_instance;
    }

    __host__ __device__ void Frontier::setSize(const ::uint64_t size) {
        this->currentSize = size;
    }

    __host__ __device__ void Frontier::insertWithoutQueueing(const ::uint64_t nodeIdx) const {
        doubleBuffer->current()[nodeIdx] = nodeIdx;
    }

    __host__ __device__ void Frontier::atomicInsertAndGrow(const ::uint64_t nodeIdx) {
        // Insert atomically in queue
        if (const auto wasInQueue{isInQueue[nodeIdx].exchange(true, cuda::memory_order_relaxed)}; !wasInQueue) {
            // Grow queue size atomically
            const auto oldSize{alternateSize.fetch_add(1, cuda::memory_order_relaxed)};

            // Insert node into next frontier
            doubleBuffer->alternate()[oldSize] = nodeIdx;
        }
    }

    __host__ __device__ void Frontier::swapToQueue() {
        // Swap buffers
        this->doubleBuffer->swap();

        // Swap sizes
        this->currentSize = alternateSize.load(cuda::memory_order_relaxed);
        this->alternateSize.store(0, cuda::memory_order_relaxed);

        // Clear isInQueue
        ::memset(isInQueue, false, doubleBuffer->getSize() * sizeof(cuda::atomic<bool, cuda::thread_scope_device>));
    }

    __host__ void Frontier::swapToQueueSync() const {
        // Swap buffers
        this->doubleBuffer->swapSync();

        if (d_instance) {
            // Get device instance
            Frontier deviceFrontier{};
            KernelUtils::cudaMemcpy(&deviceFrontier, d_instance, sizeof(Frontier), ::cudaMemcpyDeviceToHost);

            // Swap sizes
            deviceFrontier.currentSize = deviceFrontier.alternateSize.load(cuda::memory_order_relaxed);
            deviceFrontier.alternateSize.store(0, cuda::memory_order_relaxed);

            // Clear isInQueue
            KernelUtils::cudaMemset(deviceFrontier.isInQueue, false, doubleBuffer->getSize() * sizeof(cuda::atomic<bool, cuda::thread_scope_device>));

            // Update device instance
            KernelUtils::cudaMemcpy(d_instance, &deviceFrontier, sizeof(Frontier), ::cudaMemcpyHostToDevice);
        }
    }

    __host__ __device__ void Frontier::empty() {
        // Clear (virtually) the buffers
        this->currentSize = 0;
        this->alternateSize.store(0, cuda::memory_order_relaxed);

        // Clear isInQueue
        ::memset(isInQueue, false, doubleBuffer->getSize() * sizeof(cuda::atomic<bool, cuda::thread_scope_device>));
    }

    __host__ void Frontier::emptySync() const {
        if (d_instance) {
            // Get device instance
            Frontier deviceFrontier{};
            KernelUtils::cudaMemcpy(&deviceFrontier, d_instance, sizeof(Frontier), ::cudaMemcpyDeviceToHost);

            // Clear (virtually) the device buffers
            deviceFrontier.currentSize = 0;
            deviceFrontier.alternateSize.store(0, cuda::memory_order_relaxed);

            // Clear isInQueue
            KernelUtils::cudaMemset(deviceFrontier.isInQueue, 0, doubleBuffer->getSize() * sizeof(cuda::atomic<bool, cuda::thread_scope_device>));

            // Update device instance
            KernelUtils::cudaMemcpy(d_instance, &deviceFrontier, sizeof(Frontier), ::cudaMemcpyHostToDevice);
        }
    }

    __host__ __device__ Frontier::Frontier(const ::uint64_t currentSize, const ::uint64_t alternateSize, DoubleBuffer<::uint64_t>* doubleBuffer, cuda::atomic<bool, cuda::thread_scope_device>* isInQueue, Frontier* const d_instance) : currentSize(currentSize), alternateSize({alternateSize}), doubleBuffer(doubleBuffer), isInQueue(isInQueue), d_instance(d_instance) {}
} //cuSGA