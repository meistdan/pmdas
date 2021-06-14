#include <sutil/vec_math.h>
#include <optix_types.h>
#include <cuda/helpers.h>
#include <cuda/random.h>

#include <stack>
#include <vector>
#include <algorithm>
#include <iostream>
#include "kdtree.h"

namespace mdas {

#define ENABLE_VALIDATION 0

#define STACK_SIZE              64          // Size of the traversal stack in local memory
#define DYNAMIC_FETCH_THRESHOLD 20          // If fewer than this active, fetch new rays

#define divCeil(a, b) (((a) + (b) - 1) / (b))
#define roundUp(a, b) (divCeil(a, b) * (b))

    __device__ int g_warpCounter0;
    __device__ int g_warpCounter1;
    __device__ float g_error;

    texture<float4, 1> t_nodes;

    enum {
        MaxBlockHeight = 6,                     // Upper bound for blockDim.y
        EntrypointSentinel = 0x76543210,        // Bottom-most stack entry, indicating the end of traversal
    };

    template KDTree<Point3>::KDTree(
        int maxSamples,
        int candidatesNum,
        int bitsPerDim,
        int extraImgBits,
        float errorThreshold,
        float scaleX,
        float scaleY,
        std::ofstream* log
    );
    template KDTree<Point4>::KDTree(
        int maxSamples,
        int candidatesNum,
        int bitsPerDim,
        int extraImgBits,
        float errorThreshold,
        float scaleX,
        float scaleY,
        std::ofstream* log
    );
    template KDTree<Point5>::KDTree(
        int maxSamples,
        int candidatesNum,
        int bitsPerDim,
        int extraImgBits,
        float errorThreshold,
        float scaleX,
        float scaleY,
        std::ofstream* log
    );
    template KDTree<Point6>::KDTree(
        int maxSamples,
        int candidatesNum,
        int bitsPerDim,
        int extraImgBits,
        float errorThreshold,
        float scaleX,
        float scaleY,
        std::ofstream* log
    );

    template void KDTree<Point3>::InitialSampling(void);
    template void KDTree<Point4>::InitialSampling(void);
    template void KDTree<Point5>::InitialSampling(void);
    template void KDTree<Point6>::InitialSampling(void);

    template void KDTree<Point3>::ComputeErrors(void);
    template void KDTree<Point4>::ComputeErrors(void);
    template void KDTree<Point5>::ComputeErrors(void);
    template void KDTree<Point6>::ComputeErrors(void);

    template void KDTree<Point3>::AdaptiveSampling(void);
    template void KDTree<Point4>::AdaptiveSampling(void);
    template void KDTree<Point5>::AdaptiveSampling(void);
    template void KDTree<Point6>::AdaptiveSampling(void);

    template void KDTree<Point3>::SamplingPass(void);
    template void KDTree<Point4>::SamplingPass(void);
    template void KDTree<Point5>::SamplingPass(void);
    template void KDTree<Point6>::SamplingPass(void);

    template void KDTree<Point3>::Integrate(float4* pixels, uchar4* pixelsBytes, int width, int height);
    template void KDTree<Point4>::Integrate(float4* pixels, uchar4* pixelsBytes, int width, int height);
    template void KDTree<Point5>::Integrate(float4* pixels, uchar4* pixelsBytes, int width, int height);
    template void KDTree<Point6>::Integrate(float4* pixels, uchar4* pixelsBytes, int width, int height);

    template void KDTree<Point3>::SamplingDensity(float4* pixels, int width, int height);
    template void KDTree<Point4>::SamplingDensity(float4* pixels, int width, int height);
    template void KDTree<Point5>::SamplingDensity(float4* pixels, int width, int height);
    template void KDTree<Point6>::SamplingDensity(float4* pixels, int width, int height);

    template bool KDTree<Point3>::Validate(void);
    template bool KDTree<Point4>::Validate(void);
    template bool KDTree<Point5>::Validate(void);
    template bool KDTree<Point6>::Validate(void);

    float bitsToFloat(int val) {
        return *(float*)&val;
    }

    int floatToBits(float val) {
        return *(int*)&val;
    }

    template <class T>
    __device__ __inline__ void swap(T& a, T& b) {
        T t = a;
        a = b;
        b = t;
    }

    template <typename Point>
    __global__ void uniformSamplingKernel(
        int numberOfNodes,
        int samplesPerNode,
        int bitsPerDim,
        int extraImgBits,
        float scaleX,
        float scaleY,
        float* nodeErrors,
        Point* sampleCoordinates,
        KDTree<Point>::Node* nodes,
        AABB<Point>* nodeBoxes,
        unsigned int* seeds
    ) {

        // Node index
        const int nodeIndex = blockDim.x * blockIdx.x + threadIdx.x;

        if (nodeIndex < numberOfNodes) {

            // Cell offset and extent
            Point offset;
            Point extent;
            for (int j = 0; j < Point::DIM; ++j) {
                unsigned int xq = 0;
                unsigned int extentInv = 1 << bitsPerDim;
                for (int k = 0; k < bitsPerDim; ++k) {
                    int i = Point::DIM * k + (Point::DIM - j - 1);
                    xq |= ((nodeIndex >> i) & 1) << k;
                }
                if (j < 2) {
                    for (int k = bitsPerDim; k < bitsPerDim + extraImgBits; ++k) {
                        int i = Point::DIM * bitsPerDim + 2 * (k - bitsPerDim) + 1 - j;
                        xq |= ((nodeIndex >> i) & 1) << k;
                    }
                    extentInv <<= extraImgBits;
                }
                extent[j] = 1.0f / float(extentInv);
                offset[j] = xq * extent[j];
            }
            offset[0] *= scaleX;
            offset[1] *= scaleY;
            extent[0] *= scaleX;
            extent[1] *= scaleY;

            // Uniform sampling
            typename KDTree<Point>::Node node;
            unsigned int seed = tea<4>(nodeIndex, 0);
            for (int j = 0; j < samplesPerNode; ++j) {

                // Random point
                Point r;
                for (int i = 0; i < Point::DIM; ++i)
                    r.data[i] = rnd(seed);

                // Sample index
                int sampleIndex = samplesPerNode * nodeIndex + j;

                // Transform sample to the cell extent
                sampleCoordinates[sampleIndex] = offset + r * extent;

                // Sample index
                node.indices[j] = sampleIndex;

            }
            seeds[nodeIndex] = seed;

            // Node index and error
            nodeErrors[nodeIndex] = -1.0f;

            // Write node
            nodes[nodeIndex] = node;

            // Write box
            AABB<Point> box;
            box.mn = offset;
            box.mx = offset + extent;
            nodeBoxes[nodeIndex] = box;

        }

    }

    template <typename Point>
    __global__ void computeErrorsKernel(
        int numberOfLeaves,
        float* nodeErrors,
        AABB<Point>* nodeBoxes,
        float3* sampleValues
    ) {

        // Node index
        const int nodeIndex = blockDim.x * blockIdx.x + threadIdx.x;

        // Warp thread index
        const int warpThreadIndex = threadIdx.x & 31;

        // Error
        float error = 0.0f;

        if (nodeIndex < numberOfLeaves) {

            // Error
            error = nodeErrors[nodeIndex];

            // Only for new nodes
            if (error < 0.0f) {

                // Node
                float4 tmp = tex1Dfetch(t_nodes, nodeIndex);
                const typename KDTree<Point>::Node node = *(typename KDTree<Point>::Node*) & tmp;

                // Volume
                AABB<Point> box = nodeBoxes[nodeIndex];
                float volume = box.Volume();

                // Average value
                float3 avgValue = make_float3(0.0f);
                int sampleCount = 0;
                float3 sampleValuesLoc[4];
                for (int i = 0; i < 4; ++i) {
                    if (node.indices[i] >= 0) {
                        sampleValuesLoc[i] = sampleValues[node.indices[i]];
                        avgValue += sampleValuesLoc[i];
                        sampleCount++;
                    }
                }
                avgValue /= float(sampleCount);

                // Sum of differences
                float3 diffSum = make_float3(0.0f);
                for (int i = 0; i < sampleCount; ++i) {
                    float3 sampleValue = sampleValuesLoc[i];
                    diffSum.x += fabs(sampleValue.x - avgValue.x);
                    diffSum.y += fabs(sampleValue.y - avgValue.y);
                    diffSum.z += fabs(sampleValue.z - avgValue.z);
                }

                // Error
                error = 0.0f;
                if (avgValue.x != 0.0f) error += diffSum.x / avgValue.x;
                if (avgValue.y != 0.0f) error += diffSum.y / avgValue.y;
                if (avgValue.z != 0.0f) error += diffSum.z / avgValue.z;
                error /= float(sampleCount);
                error += 1.0e-5f;
                error *= volume;


                // Write error
                nodeErrors[nodeIndex] = error;

            }

        }

        // Warp-wide reduction
        float maxError = error;
        maxError = fmax(maxError, __shfl_xor_sync(0xffffffff, maxError, 1));
        maxError = fmax(maxError, __shfl_xor_sync(0xffffffff, maxError, 2));
        maxError = fmax(maxError, __shfl_xor_sync(0xffffffff, maxError, 4));
        maxError = fmax(maxError, __shfl_xor_sync(0xffffffff, maxError, 8));
        maxError = fmax(maxError, __shfl_xor_sync(0xffffffff, maxError, 16));

        // Max error
        if (warpThreadIndex == 0)
            atomicMax((int*)&g_error, __float_as_int(maxError));

    }

    template <typename Point>
    __global__ void adaptiveSamplingKernel(
        int numberOfNodes,
        int numberOfSamples,
        int maxLeafSize,
        int candidatesNum,
        float errorThreshold,
        float* nodeErrors,
        KDTree<Point>::Node* nodes,
        AABB<Point>* nodeBoxes,
        Point* sampleCoordinates,
        unsigned int* seeds
    ) {

        // Node index
        const int nodeIndex = blockDim.x * blockIdx.x + threadIdx.x;

        // Warp thread index
        const int warpThreadIndex = threadIdx.x & 31;

        // Sample indices local
        int sampleIndicesLoc[5];

        if (nodeIndex < numberOfNodes) {

            // Error
            float error = nodeErrors[nodeIndex];

            // Box
            float4 tmp;
            AABB<Point> box = nodeBoxes[nodeIndex];

            if (error >= errorThreshold * g_error) {

                // Node
                tmp = tex1Dfetch(t_nodes, nodeIndex);
                typename KDTree<Point>::Node node = *(typename KDTree<Point>::Node*) & tmp;

                // Best candidate method
                float maxDistance = -1.0;
                Point maxCandidate;
                Point diag = box.Diagonal();
                unsigned int seed = seeds[nodeIndex];
                if (seed == 0) seed = tea<4>(nodeIndex, 0);
                for (int j = 0; j < candidatesNum; ++j) {

                    // Generate candidate
                    Point r;
                    for (int i = 0; i < Point::DIM; ++i)
                        r.data[i] = rnd(seed);
                    Point candidate = box.mn + r * diag;

                    // Test samples in the leaf
                    float minDistance = FLT_MAX;
                    for (int i = 0; i < 4; ++i) {
                        if (node.indices[i] >= 0) {
                            float distance = Point::Distance(candidate, sampleCoordinates[node.indices[i]]);
                            if (minDistance > distance) {
                                minDistance = distance;
                            }
                        }
                    }

                    // Distance to the nearest neighbor
                    if (maxDistance < minDistance) {
                        maxDistance = minDistance;
                        maxCandidate = candidate;
                    }

                }
                seeds[nodeIndex] = seed;

                // Sample index
                int sampleIndex;
                {
                    // Prefix scan
                    const unsigned int activeMask = __activemask();
                    const int warpCount = __popc(activeMask);
                    const int warpIndex = __popc(activeMask & ((1u << warpThreadIndex) - 1));
                    const int warpLeader = __ffs(activeMask) - 1;

                    // Atomically add to global counter and exchange the offset
                    int warpOffset;
                    if (warpThreadIndex == warpLeader)
                        warpOffset = atomicAdd(&g_warpCounter0, warpCount);
                    warpOffset = __shfl_sync(activeMask, warpOffset, warpLeader);
                    sampleIndex = numberOfSamples + warpOffset + warpIndex;
                }

                // Sample coordinates
                sampleCoordinates[sampleIndex] = maxCandidate;

                // Sample indices
                int sampleCount = 0;
                for (int i = 0; i < 4; ++i) {
                    if (node.indices[i] >= 0) {
                        sampleIndicesLoc[i] = node.indices[i];
                        sampleCount++;
                    }
                }

                // Enough space for new sample => Just insert sample index
                if (sampleCount < maxLeafSize) {
                    node.indices[sampleCount] = sampleIndex;
                    nodes[nodeIndex] = node;
                }

                // Leaf is full => Split
                else {

                    // New sample index
                    sampleIndicesLoc[sampleCount++] = sampleIndex;

                    // Split dimension
                    int splitDimension = box.LargestAxis();
                    node.dimension = ~splitDimension;

                    // Sort 
                    for (int i = 0; i < sampleCount - 1; i++) {
                        for (int j = 0; j < sampleCount - i - 1; j++) {
                            int a = sampleIndicesLoc[j];
                            int b = sampleIndicesLoc[j + 1];
                            if (sampleCoordinates[a][splitDimension] > sampleCoordinates[b][splitDimension]) {
                                swap(sampleIndicesLoc[j], sampleIndicesLoc[j + 1]);
                            }
                        }
                    }

                    // Split position
                    int md = sampleCount >> 1;
                    float splitPosition = 0.5f * (sampleCoordinates[sampleIndicesLoc[md - 1]][splitDimension] +
                        sampleCoordinates[sampleIndicesLoc[md]][splitDimension]);
                    node.position = splitPosition;

                    // Node offset
                    int nodeOffset;
                    {
                        // Prefix scan
                        const unsigned int activeMask = __activemask();
                        const int warpCount = __popc(activeMask);
                        const int warpIndex = __popc(activeMask & ((1u << warpThreadIndex) - 1));
                        const int warpLeader = __ffs(activeMask) - 1;

                        // Atomically add to global counter and exchange the offset
                        int warpOffset;
                        if (warpThreadIndex == warpLeader)
                            warpOffset = atomicAdd(&g_warpCounter1, warpCount);
                        warpOffset = __shfl_sync(activeMask, warpOffset, warpLeader);
                        nodeOffset = numberOfNodes + warpOffset + warpIndex;
                    }

                    // Child indices
                    node.left = nodeIndex;
                    node.right = nodeOffset;

                    // Left child
                    typename KDTree<Point>::Node left;
                    for (int i = 0; i < 4; ++i) {
                        if (i < md) left.indices[i] = sampleIndicesLoc[i];
                        else left.indices[i] = ~Point::DIM;
                    }
                    nodes[node.left] = left;

                    // Left box
                    AABB<Point> childBox = box;
                    childBox.mx[splitDimension] = splitPosition;
                    nodeBoxes[node.left] = childBox;

                    // Right child
                    typename KDTree<Point>::Node right;
                    for (int i = 0; i < 4; ++i) {
                        if (i < sampleCount - md) right.indices[i] = sampleIndicesLoc[md + i];
                        else right.indices[i] = ~Point::DIM;
                    }
                    nodes[node.right] = right;

                    // Right box
                    childBox = box;
                    childBox.mn[splitDimension] = splitPosition;
                    nodeBoxes[node.right] = childBox;

                    // Reset errors
                    nodeErrors[node.left] = -1.0f;
                    nodeErrors[node.right] = -1.0f;

                }

            }

        }

    }

    template <typename Point>
    __global__ void integrateKernel(
        int numberOfNodes,
        int width,
        int height,
        float scaleX,
        float scaleY,
        float3* sampleValues,
        float4* pixels,
        AABB<Point>* nodeBoxes
    ) {

        // Node index
        const int nodeIndex = blockDim.x * blockIdx.x + threadIdx.x;

        if (nodeIndex < numberOfNodes) {

            // Pixel step
            float2 pixelStep;
            pixelStep.x = scaleX / width;
            pixelStep.y = scaleY / height;

            // Node
            float4 tmp = tex1Dfetch(t_nodes, nodeIndex);
            const typename KDTree<Point>::Node leaf = *(typename KDTree<Point>::Node*) & tmp;

            // Average value
            float3 sampleValue = make_float3(0.0f);
            int sampleCount = 0;
            for (int i = 0; i < 4; ++i) {
                if (leaf.indices[i] >= 0) {
                    sampleValue += sampleValues[leaf.indices[i]];
                    sampleCount++;
                }
            }
            sampleValue /= float(sampleCount);

            // Leaf box
            AABB<Point> box = nodeBoxes[nodeIndex];

            // Pixels covered by node's bounding box
            int x0 = box.mn[0] * width / scaleX;
            int x1 = box.mx[0] * width / scaleX;
            int y0 = box.mn[1] * height / scaleY;
            int y1 = box.mx[1] * height / scaleY;
            x1 = min(x1 + 1, width);
            y1 = min(y1 + 1, height);

            // Pixel area
            float pixArea = (scaleX * scaleY) / float(width * height);

            // Pixel bounding box
            float pMinX, pMaxX, pMinY, pMaxY;
            pMinY = y0 * scaleY / float(height);
            pMaxY = pMinY + pixelStep.y;

            // Splatting
            for (int y = y0; y < y1; ++y) {

                pMinX = x0 * scaleX / float(width);
                pMaxX = pMinX + pixelStep.x;

                for (int x = x0; x < x1; ++x) {

                    // Intersect the pixel volume with the leaf
                    const float clox = fmax(pMinX, box.mn[0]);
                    const float chix = fmin(pMaxX, box.mx[0]);
                    const float cloy = fmax(pMinY, box.mn[1]);
                    const float chiy = fmin(pMaxY, box.mx[1]);

                    if (clox < chix && cloy < chiy) {

                        // Volume
                        float volume = (chix - clox) * (chiy - cloy) * box.Volume()
                            / ((box.mx[0] - box.mn[0]) * (box.mx[1] - box.mn[1]));

                        // Add contribution
                        int pixIndex = y * width + x;
                        float3 value = sampleValue * volume / pixArea;
                        atomicAdd(&pixels[pixIndex].x, value.x);
                        atomicAdd(&pixels[pixIndex].y, value.y);
                        atomicAdd(&pixels[pixIndex].z, value.z);

                    }

                    // Update pixel bounds
                    pMinX = pMaxX;
                    pMaxX += pixelStep.x;

                }

                // Update pixel bounds
                pMinY = pMaxY;
                pMaxY += pixelStep.y;

            }

        }

    }

    template <typename Point>
    __global__ void convertToBytesKernel(
        int numberOfPixels,
        float4* pixels,
        uchar4* pixelsBytes
    ) {

        // Pixel index
        const int pixelIndex = blockDim.x * blockIdx.x + threadIdx.x;

        if (pixelIndex < numberOfPixels) {
            pixelsBytes[pixelIndex] = make_color(pixels[pixelIndex]);
        }

    }

    template <typename Point>
    KDTree<Point>::KDTree(
        int maxSamples, 
        int candidatesNum,
        int bitsPerDim,
        int extraImgBits,
        float errorThreshold, 
        float scaleX, 
        float scaleY, 
        std::ofstream* log
    ) :
        maxLeafSize(4),
        candidatesNum(candidatesNum),
        bitsPerDim(bitsPerDim),
        extraImgBits(extraImgBits),
        numberOfSamples(0),
        numberOfNodes(0),
        maxSamples(maxSamples),
        scaleX(scaleX),
        scaleY(scaleY),
        errorThreshold(errorThreshold),
        logStats(false),
        log(log)
    {
        seeds.Resize(maxSamples);
        sampleCoordinates.Resize(roundUp(maxSamples, OPTIX_ACCEL_BUFFER_BYTE_ALIGNMENT));
        sampleValues.Resize(roundUp(maxSamples, OPTIX_ACCEL_BUFFER_BYTE_ALIGNMENT));
        nodes.Resize(2 * maxSamples - 1);
        nodeBoxes.Resize(2 * maxSamples - 1);
        nodeErrors.Resize(2 * maxSamples - 1);
        if (log != nullptr) logStats = true;
        int numberOfLeaves = (1 << (bitsPerDim * Point::DIM)) << (extraImgBits << 1);
        int numberOfInitialSamples = numberOfLeaves * maxLeafSize;
        std::cout << "Initial samples " << numberOfInitialSamples << std::endl;
        std::cout << "Scale " << scaleX << " " << scaleY << std::endl;
    }

    template <typename Point>
    void KDTree<Point>::InitialSampling() {

        // Reset seeds
        cudaMemset(seeds.Data(), 0, sizeof(unsigned int) * maxSamples);

        // Number of samples
        const int samplesPerNode = 4;
        numberOfNodes = (1 << (bitsPerDim * Point::DIM)) << (extraImgBits << 1);
        numberOfSamples = numberOfNodes * samplesPerNode;

        // Grid and block size
        int minGridSize, blockSize;
        cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize,
            uniformSamplingKernel<Point>, 0, 0);
        int gridSize = divCeil(numberOfNodes, blockSize);

        // Timer
        cudaEvent_t start, stop;
        if (logStats) {
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            cudaEventRecord(start, 0);
        }

        // Launch
        uniformSamplingKernel<<<gridSize, blockSize>>>(numberOfNodes, samplesPerNode, bitsPerDim, extraImgBits, scaleX, scaleY,
            nodeErrors.Data(), sampleCoordinates.Data(), nodes.Data(), nodeBoxes.Data(), seeds.Data());

        // Elapsed time and cleanup
        if (logStats) {
            float time;
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&time, start, stop);
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
            *log << "INITIAL SAMPLES\n" << numberOfSamples << std::endl;
            *log << "INITIAL SAMPLING TIME\n" << time << std::endl;
        }

    }

    template <typename Point>
    void KDTree<Point>::ComputeErrors(void) {

        // Setup texture references
        cudaChannelFormatDesc desc = cudaCreateChannelDesc<float4>();
        CUDA_CHECK(cudaBindTexture(0, &t_nodes, nodes.Data(), &desc, sizeof(KDTree::Node) * numberOfNodes));

        // Reset atomic counter
        const float zero = 0.0f;
        cudaMemcpyToSymbol(g_error, &zero, sizeof(float));

        // Grid and block size
        int minGridSize, blockSize;
        cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize,
            computeErrorsKernel<Point>, 0, 0);
        int gridSize = divCeil(numberOfNodes, blockSize);

        // Timer
        cudaEvent_t start, stop;
        if (logStats) {
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            cudaEventRecord(start, 0);
        }

        // Launch
        computeErrorsKernel<Point><<<gridSize, blockSize>>>(numberOfNodes,
            nodeErrors.Data(), nodeBoxes.Data(), sampleValues.Data());

        // Elapsed time and cleanup
        if (logStats) {
            float time;
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&time, start, stop);
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
            *log << "COMPUTE ERRORS TIME\n" << time << std::endl;
        }

    }

    template <typename Point>
    void KDTree<Point>::AdaptiveSampling(void) {

        // Reset atomic counter
        const int zero = 0;
        cudaMemcpyToSymbol(g_warpCounter0, &zero, sizeof(int));
        cudaMemcpyToSymbol(g_warpCounter1, &zero, sizeof(int));

        // Grid and block size
        int minGridSize, blockSize;
        cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, adaptiveSamplingKernel<Point>, 0, 0);
        int gridSize = divCeil(numberOfNodes, blockSize);

        // Timer
        cudaEvent_t start, stop;
        if (logStats) {
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            cudaEventRecord(start, 0);
        }

        // Launch
        adaptiveSamplingKernel<Point><<<gridSize, blockSize>>>(
            numberOfNodes,
            numberOfSamples,
            maxLeafSize,
            candidatesNum,
            errorThreshold,
            nodeErrors.Data(),
            nodes.Data(),
            nodeBoxes.Data(),
            sampleCoordinates.Data(),
            seeds.Data()
            );

        // Number of samples
        CUDA_CHECK(cudaMemcpyFromSymbol(&newSamples, g_warpCounter0, sizeof(int), 0));
        numberOfSamples += newSamples;

        // Number of nodes
        CUDA_CHECK(cudaMemcpyFromSymbol(&newNodes, g_warpCounter1, sizeof(int), 0));
        numberOfNodes += 2 * newNodes;

        // Elapsed time and cleanup
        if (logStats) {
            float time;
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&time, start, stop);
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
            *log << "SAMPLES\n" << newSamples << std::endl;
            *log << "ADAPTIVE SAMPLING TIME\n" << time << std::endl;
        }

    }

    template <typename Point>
    void KDTree<Point>::SamplingPass(void) {
        ComputeErrors();
        AdaptiveSampling();
    }

    template <typename Point>
    void KDTree<Point>::Integrate(float4* pixels, uchar4* pixelsBytes, int width, int height) {

        // Setup texture references
        cudaChannelFormatDesc desc = cudaCreateChannelDesc<float4>();
        CUDA_CHECK(cudaBindTexture(0, &t_nodes, nodes.Data(), &desc, sizeof(KDTree::Node) * numberOfNodes));

        // Grid and block size
        int minGridSize, blockSize;
        cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize,
            integrateKernel<Point>, 0, 0);
        int gridSize = divCeil(numberOfNodes, blockSize);

        // Timer
        cudaEvent_t start, stop;
        if (logStats) {
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            cudaEventRecord(start, 0);
        }

        integrateKernel<Point><<<gridSize, blockSize>>>(numberOfNodes, width, height, scaleX, scaleY, 
            sampleValues.Data(), pixels, nodeBoxes.Data());

        // Elapsed time and cleanup
        if (logStats) {
            float time;
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&time, start, stop);
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
            *log << "TOTAL SAMPLES\n" << numberOfSamples << std::endl;
            *log << "INTEGRATE TIME\n" << time << std::endl;
        }

    }

    template <typename Point>
    void KDTree<Point>::SamplingDensity(float4* pixels, int width, int height) {
        float samplingDensity = 0.05f;
        memset(pixels, 0, sizeof(float4) * width * height);
        for (int i = 0; i < GetNumberOfSamples(); ++i) {
            int x = sampleCoordinates[i][0] / scaleX * width;
            int y = sampleCoordinates[i][1] / scaleY * height;
            if (x >= width) std::cout << "X out of bounds " << x << std::endl;
            if (y >= height) std::cout << "Y out of bounds " << y << std::endl;
            x = std::min(x, width - 1);
            y = std::min(y, height - 1);
            pixels[y * width + x] += make_float4(samplingDensity);
        }
        for (int i = 0; i < width * height; ++i)
            pixels[i].w = 1.0f;
    }

    template <typename Point>
    bool KDTree<Point>::Validate(void) {
        // Validation flag
        bool valid = true;
#if ENABLE_VALIDATION

        // Synchronize
        cudaDeviceSynchronize();

        // Volume, value, and number of samples
        float totalVolume = 0.0f;
        float3 totalValue = make_float3(0.0f);
        int totalSampleCount = 0;

        // Stack
        std::stack<int> stack;
        stack.push(0);

        // Sample index histogram
        std::vector<int> sampleHist(numberOfSamples);
        memset(sampleHist.data(), 0, sizeof(int) * numberOfSamples);

        // Find leaves and validate them
        while (!stack.empty()) {

            // Pop node index
            int nodeIndex = stack.top();
            stack.pop();

            // Box
            AABB<Point> box = nodeBoxes[nodeIndex];

            // Leaf
            if (nodes[nodeIndex].Leaf()) {

                float3 avgValue = make_float3(0.0f);
                int sampleCount = 0;
                for (int i = 0; i < 4; ++i) {
                    if (nodes[nodeIndex].indices[i] >= 0) {
                        int sampleIndex = nodes[nodeIndex].indices[i];
                        if (!box.Contains(sampleCoordinates[sampleIndex])) {
                            valid = false;
                            std::cout << "Sample is outside the leaf!" << std::endl;
                            std::cout << "Box min " << box.mn[0] << " " << box.mn[1] << " " << box.mn[2] << " " << box.mn[3] << std::endl;
                            std::cout << "Sample  " << sampleCoordinates[sampleIndex][0] << " " << sampleCoordinates[sampleIndex][1] << " "
                                << sampleCoordinates[sampleIndex][2] << " " << sampleCoordinates[sampleIndex][3] << std::endl;
                            std::cout << "Box max " << box.mx[0] << " " << box.mx[1] << " " << box.mx[2] << " " << box.mx[3] << std::endl;
                        }
                        avgValue += sampleValues[sampleIndex];
                        ++sampleHist[sampleIndex];
                        ++sampleCount;
                    }
                }
                avgValue /= float(sampleCount);

                // Add volume and sample count
                totalVolume += box.Volume();
                totalSampleCount += sampleCount;
                totalValue += avgValue * box.Volume();

            }

            // Interior
            else {
                int rightIndex = nodes[nodeIndex].right < 0 ? ~nodes[nodeIndex].right : nodes[nodeIndex].right;
                int leftIndex = nodes[nodeIndex].left < 0 ? ~nodes[nodeIndex].left : nodes[nodeIndex].left;
                stack.push(rightIndex);
                stack.push(leftIndex);
            }

        }

        float rootVolume = scaleX * scaleY;
        if (abs(totalVolume - rootVolume) > 1.0e-2 * rootVolume) {
            std::cout << "Total volume bounded by leaves is not equal to the volume of bounded by the root " <<
                totalVolume << " != " << rootVolume << std::endl;
            valid = false;
        }
        if (totalSampleCount != numberOfSamples) {
            std::cout << "Number of samples is different than number of indices in leaves " <<
                numberOfSamples << " != " << totalSampleCount << std::endl;
            valid = false;
        }
        for (int i = 0; i < numberOfSamples; ++i) {
            if (sampleHist[i] != 1) {
                valid = false;
                std::cout << "Sample not referenced or referenced more than once " << i << " " << sampleHist[i] << ": ";
                std::cout << sampleCoordinates[i][0] << " " << sampleCoordinates[i][1]
                    << " " << sampleCoordinates[i][2] << " " << sampleCoordinates[i][3] << std::endl;
                for (int k = 0; k < GetNumberOfLeaves(); ++k) {
                    for (int j = 0; j < 4; ++j) {
                        int nodeIndex = leafIndices[k];
                        KDTree::Node curNode = nodes[nodeIndex];
                        if (curNode.indices[j] == i) {
                            AABB<Point> box = nodeBoxes[nodeIndex];
                            std::cout << "Sample is in leaf node " << nodeIndex << std::endl;
                            std::cout << "\t" << box.mn[0] << " " << box.mn[1] << " " << box.mn[2] << " " << box.mn[3] << std::endl;
                            std::cout << "\t" << box.mx[0] << " " << box.mx[1] << " " << box.mx[2] << " " << box.mx[3] << std::endl;
                        }
                    }
                }
            }
        }

        // Test traversal (splitting planes)
        for (int i = 0; i < numberOfSamples; ++i) {

            // Find leaf
            stack.push(0);
            Point sample = sampleCoordinates[i];
            bool contains = false;
            while (!stack.empty()) {
                int curNodeIndex = stack.top();
                KDTree::Node curNode = nodes[curNodeIndex];
                stack.pop();
                if (curNode.Leaf()) {
                    for (int j = 0; j < 4; ++j) {
                        if (curNode.indices[j] == i)
                            contains = true;
                    }
                }
                else {
                    if (sample[~curNode.dimension] <= curNode.position)
                        stack.push(curNode.left < 0 ? ~curNode.left : curNode.left);
                    if (sample[~curNode.dimension] >= curNode.position)
                        stack.push(curNode.right < 0 ? ~curNode.right : curNode.right);
                }
            }

            if (!contains) {
                valid = false;
                std::cout << "Sample is not in any leaf " << i << ": ";
                std::cout << sampleCoordinates[i][0] << " " << sampleCoordinates[i][1] <<
                    " " << sampleCoordinates[i][2] << " " << sampleCoordinates[i][3] << std::endl;
                std::cout << "Histogram " << sampleHist[i] << std::endl;
                for (int k = 0; k < GetNumberOfLeaves(); ++k) {
                    for (int j = 0; j < 4; ++j) {
                        int nodeIndex = leafIndices[k];
                        KDTree::Node curNode = nodes[nodeIndex];
                        if (curNode.indices[j] == i) {
                            contains = true;
                            AABB<Point> box = nodeBoxes[nodeIndex];
                            std::cout << "Sample is in leaf node " << nodeIndex << std::endl;
                            std::cout << "\t" << box.mn[0] << " " << box.mn[1] << " " << box.mn[2] << " " << box.mn[3] << std::endl;
                            std::cout << "\t" << box.mx[0] << " " << box.mx[1] << " " << box.mx[2] << " " << box.mx[3] << std::endl;
                        }
                    }
                }
            }
        }

        if (!valid) exit(1);
#endif
        return valid;
    }

}  // namespace mdas