#include <sutil/vec_math.h>
#include <cuda/helpers.h>
#include <cuda/random.h>

#include <stack>
#include <vector>
#include <algorithm>
#include <iostream>
#include "kdtree.h"

namespace mdas {

#define STACK_SIZE              64          // Size of the traversal stack in local memory.
#define DYNAMIC_FETCH_THRESHOLD 20          // If fewer than this active, fetch new rays

#define divCeil(a, b) (((a) + (b) - 1) / (b))

__device__ int g_warpCounter0;
__device__ int g_warpCounter1;
__device__ float g_error;

texture<float4, 1> t_nodes;
texture<float4, 1> t_nodeBoxes;

enum {
    MaxBlockHeight = 6,                     // Upper bound for blockDim.y
    EntrypointSentinel = 0x76543210,        // Bottom-most stack entry, indicating the end of traversal
};

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

__global__ void uniformSamplingKernel(
    int numberOfLeaves,
    int samplesPerLeaf,
    int bitsPerDim,
    int extraImgBits,
    float scaleX,
    float scaleY,
    float* nodeErrors,
    int* leafIndices,
    Point* sampleCoordinates,
    KDTree::Node* nodes,
    AABB* nodeBoxes,
    unsigned int* seeds
) {

    // Leaf index
    const int leafIndex = blockDim.x * blockIdx.x + threadIdx.x;

    if (leafIndex < numberOfLeaves) {

        // Cell offset and extent
        Point offset;
        Point extent;
        for (int j = 0; j < Point::DIM; ++j) {
            unsigned int xq = 0;
            unsigned int extentInv = 1 << bitsPerDim;
            for (int k = 0; k < bitsPerDim; ++k) {
                int i = Point::DIM * k + (Point::DIM - j - 1);
                xq |= ((leafIndex >> i) & 1) << k;
            }
            if (j < 2) {
                for (int k = bitsPerDim; k < bitsPerDim + extraImgBits; ++k) {
                    int i = Point::DIM * bitsPerDim + 2 * (k - bitsPerDim) + 1 - j;
                    xq |= ((leafIndex >> i) & 1) << k;
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
        KDTree::Node node;
        unsigned int seed = tea<4>(leafIndex, 0);
        for (int j = 0; j < samplesPerLeaf; ++j) {

            // Random point
            Point r;
            for (int i = 0; i < Point::DIM; ++i) {
                r.data[i] = rnd(seed);
            }

            // Sample index
            int sampleIndex = samplesPerLeaf * leafIndex + j;

            // Transform sample to the cell extent
            sampleCoordinates[sampleIndex] = offset + r * extent;

            // Sample index
            node.indices[j] = sampleIndex;

        }
        seeds[leafIndex] = seed;
        
        // Node index and error
        int nodeIndex = leafIndex + numberOfLeaves - 1;
        leafIndices[leafIndex] = nodeIndex;
        nodeErrors[nodeIndex] = -1.0f;

        // Write node
        nodes[nodeIndex] = node;

        // Write box
        AABB box;
        box.mn = offset;
        box.mx = offset + extent;
        nodeBoxes[nodeIndex] = box;

    }

}

__global__ void constructKernel(
    int numberOfInteriors,
    int maxLeafSize,
    int bitsPerDim,
    int extraImgBits,
    float scaleX,
    float scaleY,
    KDTree::Node* nodes
) {

    // Node index.
    const int nodeIndex = blockDim.x * blockIdx.x + threadIdx.x;

    if (nodeIndex < numberOfInteriors) {

        // Split dimension
        const int bit = 8 * sizeof(unsigned int) - __clz(unsigned int(nodeIndex + 1)) - 1;
        int dimension = bit < 2 * extraImgBits ? bit & 1 : (bit - 2 * extraImgBits) % Point::DIM;

        // Split position
        unsigned int c = nodeIndex - (1 << bit) + 1;
        float increase = 0.5f;
        float position = 0.0f;
#if 0
        for (int t = bit - dimension - 1; t >= 0; t -= Point::DIM) {
            if ((c >> t) & 1) position += increase;
            increase *= 0.5f;
        }
#else
        int delta = 2;
        int tm = bit - dimension - 1;
        int t0 = tm;
        int th = tm;
        if (dimension < 2) th -= 2 * extraImgBits;
        else t0 -= 2 * extraImgBits;
        for (int t = t0; t >= 0; t -= delta) {
            if ((c >> t) & 1) position += increase;
            if (t <= th) delta = Point::DIM;
            increase *= 0.5f;
        }
#endif
        position += increase;

        // Scale
        if (dimension == 0) position *= scaleX;
        if (dimension == 1) position *= scaleY;

        // Write node
        KDTree::Node node;
        node.right = (nodeIndex + 1) << 1;
        node.left = node.right - 1;
        node.dimension = ~dimension;
        node.position = position;
        nodes[nodeIndex] = node;

    }

}

__global__ void updateIndicesKernel(
    int numberOfNodes,
    KDTree::Node* nodes
) {

    // Sample index.
    const int nodeIndex = blockDim.x * blockIdx.x + threadIdx.x;

    if (nodeIndex < numberOfNodes) {

        // Node
        KDTree::Node node = nodes[nodeIndex];

        // Only interiors
        if (!node.Leaf()) {
            node.left = node.left < 0 ? ~node.left : node.left;
            node.right = node.right < 0 ? ~node.right : node.right;
            if (nodes[node.left].Leaf()) node.left = ~node.left;
            if (nodes[node.right].Leaf()) node.right = ~node.right;
        }

        // Write node
        nodes[nodeIndex] = node;

    }

}

__global__ void computeErrorsKernel(
    int numberOfLeaves,
    int* leafIndices,
    float* nodeErrors,
    float3* sampleValues
) {

    // Leaf index.
    const int leafIndex = blockDim.x * blockIdx.x + threadIdx.x;

    // Warp thread index.
    const int warpThreadIndex = threadIdx.x & 31;

    // Error
    float error = 0.0f;

    if (leafIndex < numberOfLeaves) {

        // Node index
        int nodeIndex = leafIndices[leafIndex];

        // Error
        error = nodeErrors[nodeIndex];

        // Only for new nodes
        if (error < 0.0f) {
            
            // Node
            float4 tmp = tex1Dfetch(t_nodes, nodeIndex);
            const KDTree::Node node = *(KDTree::Node*)&tmp;

            // Volume
            AABB box;
            tmp = tex1Dfetch(t_nodeBoxes, 2 * nodeIndex + 0); box.mn = *(Point*)&tmp;
            tmp = tex1Dfetch(t_nodeBoxes, 2 * nodeIndex + 1); box.mx = *(Point*)&tmp;
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
            error += 1.0e-5;
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

__global__ void adaptiveSamplingKernel(
    int numberOfLeaves,
    int numberOfNodes,
    int numberOfSamples,
    int maxLeafSize,
    int candidatesNum,
    float errorThreshold,
    float scaleX,
    float scaleY,
    int* leafIndices,
    float* nodeErrors,
    KDTree::Node* nodes,
    AABB* nodeBoxes,
    Point* sampleCoordinates,
    unsigned int* seeds
) {

    // Leaf index.
    const int leafIndex = blockDim.x * blockIdx.x + threadIdx.x;

    // Warp thread index.
    const int warpThreadIndex = threadIdx.x & 31;

    // Stack
    int stack[STACK_SIZE];
    stack[0] = EntrypointSentinel;

    if (leafIndex < numberOfLeaves) {

        // Node index
        int nodeIndex = leafIndices[leafIndex];

        // Error
        float error = nodeErrors[nodeIndex];

        // Box
        AABB box;
        float4 tmp;
        tmp = tex1Dfetch(t_nodeBoxes, 2 * nodeIndex + 0); box.mn = *(Point*)&tmp;
        tmp = tex1Dfetch(t_nodeBoxes, 2 * nodeIndex + 1); box.mx = *(Point*)&tmp;

        // Radius and center
        const float R = 0.55f;
        float radius = R * Point::Distance(box.mx, box.mn);
        Point center = box.Center();

        // Setup traversal
        int* stackPtr = &stack[0];
        int curNodeIndex = 0;

        while (curNodeIndex != EntrypointSentinel) {

            // Node
            tmp = tex1Dfetch(t_nodes, curNodeIndex);
            KDTree::Node node = *(KDTree::Node*) & tmp;

            // Leaf
            if (node.Leaf()) {

                // Update error
                error = max(error, nodeErrors[curNodeIndex]);

                // Pop
                curNodeIndex = *stackPtr;
                --stackPtr;

            }

            // Interior
            else {

                bool traverseChild0 = center[node.dimension] - radius <= node.position;
                bool traverseChild1 = center[node.dimension] + radius >= node.position;

                // Neither child was intersected => pop stack
                if (!traverseChild0 && !traverseChild1) {
                    curNodeIndex = *stackPtr;
                    --stackPtr;
                }

                // Otherwise => fetch child pointers
                else {
                    curNodeIndex = traverseChild0 ? node.left : node.right;

                    // Both children were intersected => push the farther one
                    if (traverseChild0 && traverseChild1) {
                        ++stackPtr;
                        *stackPtr = node.right;
                    }
                }

            }

        }

        if (error >= errorThreshold * g_error) {

            // Node
            tmp = tex1Dfetch(t_nodes, nodeIndex);
            KDTree::Node node = *(KDTree::Node*) & tmp;

            // Best candidate method
            float maxDistance = -1.0;
            Point maxCandidate;
            Point diag = box.Diagonal();
            unsigned int seed = seeds[leafIndex];
            if (seed == 0) seed = tea<4>(leafIndex, 0);
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
            seeds[leafIndex] = seed;

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
                    stack[i] = node.indices[i];
                    sampleCount++;
                }
            }

            // Enough space for new sample => Just insert sample index
            if (sampleCount < maxLeafSize) {
                node.indices[sampleCount] = sampleIndex;
            }

            // Leaf is full => Split
            else {

                // New sample index
                stack[sampleCount++] = sampleIndex;

                // Split dimension
                int splitDimension = box.LargestAxis();
                node.dimension = ~splitDimension;

                // Sort 
                for (int i = 0; i < sampleCount - 1; i++) {
                    for (int j = 0; j < sampleCount - i - 1; j++) {
                        int a = stack[j];
                        int b = stack[j + 1];
                        if (sampleCoordinates[a][splitDimension] > sampleCoordinates[b][splitDimension]) {
                            swap(stack[j], stack[j + 1]);
                        }
                    }
                }

                // Split position
                int md = sampleCount >> 1;
                float splitPosition = 0.5f * (sampleCoordinates[stack[md - 1]][splitDimension] +
                    sampleCoordinates[stack[md]][splitDimension]);
                node.position = splitPosition;

                // Node offset
                int nodeOffset;
                int leafOffset;
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
                    nodeOffset = numberOfNodes + 2 * (warpOffset + warpIndex);
                    leafOffset = numberOfLeaves + warpOffset + warpIndex;
                }

                // Child indices
                node.left = nodeOffset;
                node.right = nodeOffset + 1;

                // Left child
                KDTree::Node left;
                for (int i = 0; i < 4; ++i) {
                    if (i < md) left.indices[i] = stack[i];
                    else left.indices[i] = ~Point::DIM;
                }
                nodes[node.left] = left;

                // Left box
                AABB childBox = box;
                childBox.mx[splitDimension] = splitPosition;
                nodeBoxes[node.left] = childBox;

                // Right child
                KDTree::Node right;
                for (int i = 0; i < 4; ++i) {
                    if (i < sampleCount - md) right.indices[i] = stack[md + i];
                    else right.indices[i] = ~Point::DIM;
                }
                nodes[node.right] = right;

                // Right box
                childBox = box;
                childBox.mn[splitDimension] = splitPosition;
                nodeBoxes[node.right] = childBox;

                // New leaf indices and reset error
                leafIndices[leafIndex] = node.left;
                leafIndices[leafOffset] = node.right;

                // Reset errors
                nodeErrors[node.left] = -1.0f;
                nodeErrors[node.right] = -1.0f;

            }

            // Write node
            nodes[nodeIndex] = node;

        }

    }

}

__global__ void integrateKernel(
    int width,
    int height,
    float scaleX,
    float scaleY,
    float3* sampleValues,
    float4* pixels,
    uchar4* pixelsBytes
) {

    // Traversal stack in CUDA thread-local memory
    int traversalStack[STACK_SIZE];
    traversalStack[0] = EntrypointSentinel; // Bottom-most entry

    // Live state during traversal, stored in registers
    char*   stackPtr;                       // Current position in traversal stack
    int     leafAddr;                       // First postponed leaf, non-negative if none
    int     nodeAddr = EntrypointSentinel;  // Non-negative: current internal node, negative: second postponed leaf
    int     pixIdx;
    int     pixX, pixY;
    float   pMinX, pMaxX, pMinY, pMaxY;
    float   pixArea = (scaleX * scaleY) / (width * height);
    float3  value;

    // Initialize persistent threads.
    __shared__ volatile int nextPixArray[MaxBlockHeight]; // Current ray index in global buffer

    // Persistent threads: fetch and process rays in a loop
    do {
        const int tidx = threadIdx.x;
        volatile int& pixBase = nextPixArray[threadIdx.y];

        // Fetch new rays from the global pool using lane 0
        const bool          terminated = nodeAddr == EntrypointSentinel;
        //const unsigned int  maskTerminated = __ballot_sync(0xffffffff, terminated);
        const unsigned int  maskTerminated = __ballot_sync(__activemask(), terminated);
        const int           numTerminated = __popc(maskTerminated);
        const int           idxTerminated = __popc(maskTerminated & ((1u << tidx) - 1));

        if (terminated) {

            if (idxTerminated == 0)
                pixBase = atomicAdd(&g_warpCounter0, numTerminated);

            pixIdx = pixBase + idxTerminated;
            if (pixIdx >= width * height)
                break;

            // Value
            value = make_float3(0.0f);

            // Compute pixel
            pixX = pixIdx % width;
            pixY = pixIdx / width;
            pMinX = pixX / float(width) * scaleX;
            pMaxX = (pixX + 1) / float(width) * scaleX;
            pMinY = pixY / float(height) * scaleY;
            pMaxY = (pixY + 1) / float(height) * scaleY;

            // Setup traversal
            stackPtr = (char*)&traversalStack[0];
            leafAddr = 0;   // No postponed leaf
            nodeAddr = 0;   // Start from the root
        }

        // Traversal loop
        while (nodeAddr != EntrypointSentinel) {
            // Traverse internal nodes until all SIMD lanes have found a leaf
            while (unsigned int(nodeAddr) < unsigned int(EntrypointSentinel)) {

                // Fetch AABBs of the two child nodes
                float4 tmp = tex1Dfetch(t_nodes, nodeAddr); // child_index0, child_index1
                const KDTree::Node  node = *(KDTree::Node*)&tmp;

                // Intersect the pixel volume with the child nodes
                bool traverseChild0x = ~node.dimension == 0 ? pMinX < node.position : true;
                bool traverseChild0y = ~node.dimension == 1 ? pMinY < node.position : true;
                bool traverseChild1x = ~node.dimension == 0 ? pMaxX > node.position : true;
                bool traverseChild1y = ~node.dimension == 1 ? pMaxY > node.position : true;

                bool traverseChild0 = traverseChild0x && traverseChild0y;
                bool traverseChild1 = traverseChild1x && traverseChild1y;

                // Neither child was intersected => pop stack
                if (!traverseChild0 && !traverseChild1) {
                    nodeAddr = *(int*)stackPtr;
                    stackPtr -= 4;
                }

                // Otherwise => fetch child pointers
                else {
                    nodeAddr = (traverseChild0) ? node.left : node.right;

                    // Both children were intersected => push the farther one
                    if (traverseChild0 && traverseChild1) {
                        stackPtr += 4;
                        *(int*)stackPtr = node.right;
                    }
                }

                // First leaf => postpone and continue traversal
                if (nodeAddr < 0 && leafAddr >= 0) {
                    leafAddr = nodeAddr;
                    nodeAddr = *(int*)stackPtr;
                    stackPtr -= 4;
                }

                // All SIMD lanes have found a leaf? => process them

                // NOTE: inline PTX implementation of "if(!__any(leafAddr >= 0)) break;"
                // tried everything with CUDA 4.2 but always got several redundant instructions

                //unsigned int mask;
                //asm("{\n"
                //    "   .reg .pred p;               \n"
                //    "setp.ge.s32        p, %1, 0;   \n"
                //    "vote.ballot.b32    %0,p;       \n"
                //    "}"
                //    : "=r"(mask)
                //    : "r"(leafAddr));
                //if (!mask)
                //    break;

                if(!__any_sync(__activemask(), leafAddr >= 0))
                    break;

            }

            // Process postponed leaf nodes
            while (leafAddr < 0) {

                // Node
                float4 tmp = tex1Dfetch(t_nodes, ~leafAddr);
                const KDTree::Node leaf = *(KDTree::Node*)&tmp;

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
                AABB box;
                tmp = tex1Dfetch(t_nodeBoxes, 2 * (~leafAddr) + 0); box.mn = *(Point*)&tmp;
                tmp = tex1Dfetch(t_nodeBoxes, 2 * (~leafAddr) + 1); box.mx = *(Point*)&tmp;

                // Intersect the pixel volume with the leaf
                const float clox = fmax(pMinX, box.mn[0]);
                const float chix = fmin(pMaxX, box.mx[0]);
                const float cloy = fmax(pMinY, box.mn[1]);
                const float chiy = fmin(pMaxY, box.mx[1]);

                // Volume
                float volume = (chix - clox) * (chiy - cloy) * (box.mx[2] - box.mn[2]) * (box.mx[3] - box.mn[3]);

                // Add contribution
                value += sampleValue * volume / pixArea;

                // Another leaf was postponed => process it as well.
                leafAddr = nodeAddr;
                if (nodeAddr < 0) {
                    nodeAddr = *(int*)stackPtr;
                    stackPtr -= 4;
                }
            } // leaf

            // DYNAMIC FETCH
            if (__popc(__activemask()) < DYNAMIC_FETCH_THRESHOLD)
                break;

        } // traversal

        // Store the result
        pixels[pixIdx] = make_float4(value, 1.0f);
        pixelsBytes[pixIdx] = make_color(value);

    } while (true);

}


KDTree::KDTree(int maxSamples, std::ofstream* log) : 
    maxLeafSize(4), 
    candidatesNum(1), 
    bitsPerDim(0), 
    extraImgBits(8), 
    numberOfSamples(0), 
    numberOfNodes(0), 
    maxSamples(maxSamples), 
    scaleX(1024.0f), 
    scaleY(512.0f), 
    errorThreshold(0.025f),
    logStats(false),
    log(log)
{
    seeds.Resize(maxSamples);
    sampleCoordinates.Resize(maxSamples);
    sampleValues.Resize(maxSamples);
    nodes.Resize(2 * maxSamples - 1);
    nodeBoxes.Resize(2 * maxSamples - 1);
    nodeErrors.Resize(2 * maxSamples - 1);
    leafIndices.Resize(maxSamples);
    if (log != nullptr) logStats = true;
    int numberOfLeaves = (1 << (bitsPerDim * Point::DIM)) << (extraImgBits << 1);
    int numberOfInitialSamples = numberOfLeaves * maxLeafSize;
    std::cout << "Initial samples " << numberOfInitialSamples << std::endl;
}

void KDTree::InitialSampling() {

    // Reset seeds
    cudaMemset(seeds.Data(), 0, sizeof(unsigned int) * maxSamples);

    // Number of samples
    int numberOfLeaves = (1 << (bitsPerDim  * Point::DIM)) << (extraImgBits << 1);
    numberOfSamples = numberOfLeaves * maxLeafSize;

    // Grid and block size
    int minGridSize, blockSize;
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize,
        uniformSamplingKernel, 0, 0);
    int gridSize = divCeil(numberOfLeaves, blockSize);

    // Timer
    cudaEvent_t start, stop;
    if (logStats) {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);
    }

    // Launch
    uniformSamplingKernel<<<gridSize, blockSize>>>(numberOfLeaves, maxLeafSize, bitsPerDim, extraImgBits, scaleX, scaleY,
        nodeErrors.Data(), leafIndices.Data(), sampleCoordinates.Data(), nodes.Data(), nodeBoxes.Data(), seeds.Data());

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

void KDTree::Construct(void) {

    // Number of nodes
    int numberOfLeaves = numberOfSamples / maxLeafSize;
    numberOfNodes = 2 * numberOfLeaves - 1;

    // Grid and block size
    int minGridSize, blockSize;
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize,
        constructKernel, 0, 0);
    int gridSize = divCeil(numberOfLeaves - 1, blockSize);

    // Timer
    cudaEvent_t start, stop;
    if (logStats) {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);
    }

    // Launch
    constructKernel<<<gridSize, blockSize>>>(numberOfLeaves - 1, maxLeafSize, bitsPerDim,
        extraImgBits, scaleX, scaleY, nodes.Data());

    // Elapsed time and cleanup
    if (logStats) {
        float time;
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time, start, stop);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        *log << "CONSTRUCT TIME\n" << time << std::endl;
    }

}

void KDTree::UpdateIndices(void) {
    
    // Grid and block size
    int minGridSize, blockSize;
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize,
        updateIndicesKernel, 0, 0);
    int gridSize = divCeil(numberOfNodes, blockSize);

    // Timer
    cudaEvent_t start, stop;
    if (logStats) {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);
    }

    // Launch
    updateIndicesKernel<<<gridSize, blockSize>>>(numberOfNodes, nodes.Data());

    // Elapsed time and cleanup
    if (logStats) {
        float time;
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time, start, stop);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        *log << "UPDATE INDICES TIME\n" << time << std::endl;
    }

}

void KDTree::ComputeErrors(void) {

    // Setup texture references
    cudaChannelFormatDesc desc = cudaCreateChannelDesc<float4>();
    CUDA_CHECK(cudaBindTexture(0, &t_nodes, nodes.Data(), &desc, sizeof(KDTree::Node) * numberOfNodes));
    CUDA_CHECK(cudaBindTexture(0, &t_nodeBoxes, nodeBoxes.Data(), &desc, sizeof(AABB) * numberOfNodes));

    // Reset atomic counter
    const float zero = 0.0f;
    cudaMemcpyToSymbol(g_error, &zero, sizeof(float));

    // Grid and block size
    int minGridSize, blockSize;
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize,
        computeErrorsKernel, 0, 0);
    int gridSize = divCeil(GetNumberOfLeaves(), blockSize);

    // Timer
    cudaEvent_t start, stop;
    if (logStats) {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);
    }

    // Launch
    computeErrorsKernel<<<gridSize, blockSize>>>(GetNumberOfLeaves(), 
        leafIndices.Data(), nodeErrors.Data(), sampleValues.Data());

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


void KDTree::AdaptiveSampling(void) {

    // Reset atomic counter
    const int zero = 0;
    cudaMemcpyToSymbol(g_warpCounter0, &zero, sizeof(int));
    cudaMemcpyToSymbol(g_warpCounter1, &zero, sizeof(int));

    // Grid and block size
    int minGridSize, blockSize;
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, adaptiveSamplingKernel, 0, 0);
    int gridSize = divCeil(GetNumberOfLeaves(), blockSize);

    // Timer
    cudaEvent_t start, stop;
    if (logStats) {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);
    }

    // Launch
    adaptiveSamplingKernel<<<gridSize, blockSize>>>(
        GetNumberOfLeaves(),
        numberOfNodes,
        numberOfSamples,
        maxLeafSize,
        candidatesNum,
        errorThreshold,
        scaleX,
        scaleY,
        leafIndices.Data(),
        nodeErrors.Data(),
        nodes.Data(),
        nodeBoxes.Data(),
        sampleCoordinates.Data(),
        seeds.Data()
        );

    // Number of samples
    cudaMemcpyFromSymbol(&newSamples, g_warpCounter0, sizeof(int), 0);
    numberOfSamples += newSamples;

    // Number of nodes
    int newInteriors;
    cudaMemcpyFromSymbol(&newInteriors, g_warpCounter1, sizeof(int), 0);
    numberOfNodes += 2 * newInteriors;

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

void KDTree::Build() {
    InitialSampling();
    Construct();
}

void KDTree::SamplingPass(void) {
    ComputeErrors();
    AdaptiveSampling();
}

void KDTree::Integrate(float4* pixels, uchar4* pixelsBytes, int width, int height) {

    // Grid and block size
    const int desiredWarps = 720;
    dim3 blockSize(32, 4);
    int blockWarps = (blockSize.x * blockSize.y + 31) / 32; // 4
    int gridSize = (desiredWarps + blockWarps - 1) / blockWarps;

    // Setup texture references
    cudaChannelFormatDesc desc = cudaCreateChannelDesc<float4>();
    CUDA_CHECK(cudaBindTexture(0, &t_nodes, nodes.Data(), &desc, sizeof(KDTree::Node) * numberOfNodes));
    CUDA_CHECK(cudaBindTexture(0, &t_nodeBoxes, nodeBoxes.Data(), &desc, sizeof(AABB) * numberOfNodes));

    // Reset atomic counter
    const int zero = 0;
    cudaMemcpyToSymbol(g_warpCounter0, &zero, sizeof(int));

    // Timer
    cudaEvent_t start, stop;
    if (logStats) {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);
    }

    // Launch
    integrateKernel<<<gridSize, blockSize>>>(width, height, scaleX, scaleY, sampleValues.Data(), pixels, pixelsBytes);

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

void KDTree::SamplingDensity(float4* pixels, int width, int height) {
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

bool KDTree::Validate(void) {

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

    // Validation flag
    bool valid = true;

    // Find leaves and validate them
    while (!stack.empty()) {

        // Pop node index
        int nodeIndex = stack.top();
        stack.pop();

        // Box
        AABB box = nodeBoxes[nodeIndex];

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
            std::cout << "Sample not referenced or referenced more than once "  << i << " " << sampleHist[i] << ": ";
            std::cout << sampleCoordinates[i][0] << " " << sampleCoordinates[i][1] 
               << " " << sampleCoordinates[i][2] << " " << sampleCoordinates[i][3] << std::endl;
            for (int k = 0; k < GetNumberOfLeaves(); ++k) {
                for (int j = 0; j < 4; ++j) {
                    int nodeIndex = leafIndices[k];
                    KDTree::Node curNode = nodes[nodeIndex];
                    if (curNode.indices[j] == i) {
                        AABB box = nodeBoxes[nodeIndex];
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
                        AABB box = nodeBoxes[nodeIndex];
                        std::cout << "Sample is in leaf node " << nodeIndex << std::endl;
                        std::cout << "\t" << box.mn[0] << " " << box.mn[1] << " " << box.mn[2] << " " << box.mn[3] << std::endl;
                        std::cout << "\t" << box.mx[0] << " " << box.mx[1] << " " << box.mx[2] << " " << box.mx[3] << std::endl;
                    }
                }
            }
        }
    }

    if (!valid) exit(1);

    return valid;
}

}  // namespace mdas