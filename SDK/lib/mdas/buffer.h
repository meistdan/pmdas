#ifndef _BUFFER_H_
#define _BUFFER_H_

#include <cuda_runtime.h>

template<typename T>
bool checkCudaError_(T result, char const* const func, const char* const file, int const line) {
    if (result) {
        printf("CUDA error at %s:%d code=%d \"%s\"\n", file, line, static_cast<unsigned int>(result), func);
        return true;
    }
    else {
        return false;
    }
}
#define CUDA_CHECK(val) checkCudaError_((val),#val,__FILE__,__LINE__)


template <typename T>
class Buffer {

public:

    Buffer(void) : size(0), totalSize(0), data(nullptr) {}

    Buffer(size_t size) : size(0), totalSize(0), data(nullptr) {
        Resize(size);
    }

    Buffer(const Buffer& other) : size(other.size), totalSize(other.totalSize), data(nullptr) {
        Resize(other.totalSize);
        size = other.size;
        if (size > 0)
            CUDA_CHECK(cudaMemcpy(data, other.data, sizeof(T) * other.size, cudaMemcpyDeviceToDevice));
    }

    ~Buffer(void) {
        if (data) cudaFree(data);
    }

    void Resize(size_t size) {
        if (size > totalSize) {
            if (data) cudaFree(data);
            CUDA_CHECK(cudaMallocManaged(&data, sizeof(T) * size));
            totalSize = size;
        }
        this->size = size;
    }

    T* Data(void) {
        return data;
    }

    T& operator[](int index) {
        return data[index];
    }

    Buffer& operator=(const Buffer& other) {
        Resize(other.totalSize);
        size = other.size;
        CUDA_CHECK(cudaMemcpy(data, other.data, sizeof(T) * other.size, cudaMemcpyDeviceToDevice));
        return *this;
    }

    size_t Size(void) {
        return size;
    }

private:
    T* data;
    size_t size;
    size_t totalSize;
};

#endif /* _BUFFER_H_ */
