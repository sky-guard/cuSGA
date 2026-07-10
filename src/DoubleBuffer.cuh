#ifndef CUSGA_DOUBLEBUFFER_CUH
#define CUSGA_DOUBLEBUFFER_CUH
#include "KernelUtils.cuh"

namespace cuSGA {
    template <typename T>
    class DoubleBuffer {
    public:
        // Double buffer related constants
        static constexpr ::size_t NUM_DOUBLE_BUFFERS{2};

        // Create double buffer
        static DoubleBuffer* create(const ::size_t size) {
            // Create double buffer instance
            auto doubleBuffer{new DoubleBuffer(size)};

            return doubleBuffer;
        }

        // Move double buffer to device
        __host__ DoubleBuffer* copyToDevice() {
            // Check if device instance already exists for this double buffer
            if (d_instance) {
                return d_instance;
            }

            // Allocate device double buffer
            T* d_buffers[NUM_DOUBLE_BUFFERS]{nullptr};
            if (size > 0) {
                for (::size_t i{0}; i < NUM_DOUBLE_BUFFERS; ++i) {
                    // Allocate device buffer
                    KernelUtils::cudaMalloc(&d_buffers[i], size * sizeof(T));

                    // Copy buffer data from host to device
                    KernelUtils::cudaMemcpy(d_buffers[i], buffers[i], size * sizeof(T), ::cudaMemcpyHostToDevice);
                }
            }

            // Allocate device sequence instance
            DoubleBuffer* d_doubleBuffer{nullptr};
            KernelUtils::cudaMalloc(&d_doubleBuffer, sizeof(DoubleBuffer));

            // Create temporary host instance holding the device pointers
            const DoubleBuffer deviceDoubleBuffer{size, d_buffers[0], d_buffers[1], d_doubleBuffer};

            // Update host instance data
            this->d_instance = d_doubleBuffer;

            // Copy instance data from host to device
            KernelUtils::cudaMemcpy(d_doubleBuffer, &deviceDoubleBuffer, sizeof(DoubleBuffer), ::cudaMemcpyHostToDevice);

            return d_doubleBuffer;
        }

        // Free double buffer
        __host__ void free() const {
            // Free device memory if device instance is present
            if (d_instance) {
                // Create a temporary host copy of the device instance to get its internal device pointers
                DoubleBuffer deviceDoubleBuffer{};
                KernelUtils::cudaMemcpy(&deviceDoubleBuffer, d_instance, sizeof(DoubleBuffer), ::cudaMemcpyDeviceToHost);

                // Free device double buffer
                for (auto& buffer : deviceDoubleBuffer.buffers) {
                    if (buffer) {
                        KernelUtils::cudaFree(buffer);
                    }
                }

                // Free device double buffer instance
                KernelUtils::cudaFree(d_instance);
            }

            // Free host memory
            for (::size_t i{0}; i < NUM_DOUBLE_BUFFERS; ++i) {
                delete[] buffers[i];
            }
            delete this;
        }

        // Get size
        __host__ __device__ ::size_t getSize() const {
            return size;
        }

        // Get current buffer
        __host__ __device__ T* current() const {
            // Get current buffer
            return buffers[selector];
        }

        // Get alternate buffer
        __host__ __device__ T* alternate() const {
            // Get alternate buffer
            return buffers[selector ^ 1];
        }

        // Get device instance
        __host__ __device__ DoubleBuffer* getDeviceInstance() const {
            return d_instance;
        }

        // Swap buffers
        __host__ __device__ void swap() {
            this->selector ^= 1;
        }

        // Swap device buffers
        __host__ void swapSync() {
            if (d_instance) {
                this->d_selector ^= 1;
                KernelUtils::cudaMemcpy(&d_instance->selector, &d_selector, sizeof(selector), ::cudaMemcpyHostToDevice);
            }
        }

        // Non-const version of subscription operator for assignment and modification
        __host__ __device__ T& operator[](const ::size_t idx) {
            return buffers[selector][idx];
        }

        // Const version of subscription operator for read-only access
        __host__ __device__ const T& operator[](const ::size_t idx) const {
            return buffers[selector][idx];
        }

    private:
        // Double buffer implementation
        ::size_t size{0};
        T* buffers[NUM_DOUBLE_BUFFERS]{nullptr};
        ::size_t selector{0};
        ::size_t d_selector{0};
        DoubleBuffer* d_instance{nullptr};

        // Default constructor
        DoubleBuffer() = default;

        // Double buffer constructor
        __host__ __device__ explicit DoubleBuffer(const ::size_t size, T* const currentBuffer = nullptr, T* const alternateBuffer = nullptr, DoubleBuffer* const d_instance = nullptr) : size(size), buffers{currentBuffer, alternateBuffer}, d_instance(d_instance) {
            // Allocate buffers if missing
            if (size > 0) {
                if (!currentBuffer) {
                    buffers[0] = new T[size]{};
                }
                if (!alternateBuffer) {
                    buffers[1] = new T[size]{};
                }
            }
        }
    };
} // cuSGA

#endif //CUSGA_DOUBLEBUFFER_CUH
