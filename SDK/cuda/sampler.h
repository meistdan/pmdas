#pragma once
#include "random.h"

class Sampler {
public:

    __host__ __device__ Sampler(unsigned int seed) : cur_dim(0), cur_index(0) {
        for (int i = 0; i < 6; ++i)
            offsets[i] = rnd(seed);
    }

    __host__ __device__ __inline__ float get() { 
        float sample = halton(cur_dim, cur_index) + offsets[cur_dim];
        ++cur_dim;
        return sample;
    }

    __host__ __device__ __inline__ void next_sample() { 
        cur_dim = 0;
        cur_index++;
    }

private:
    float offsets[6];
    int cur_index;
    int cur_dim;
};
