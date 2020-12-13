#ifndef _KDTREE_H_
#define _KDTREE_H_

#include "buffer.h"
#include "point.h"
#include <fstream>

namespace mdas {

class KDTree {

public:

    struct Node {
        union {
            struct {
                int indices[4];
            };
            struct {
                int left;
                int right;
                int dimension;
                float position;
            };
        };
        __host__ __device__ Node(void) {
            for (int i = 0; i < 4; ++i)
                indices[i] = ~Point::DIM;
        }
        __host__ __device__ bool Leaf(void) {
            return dimension >= 0 || dimension <= ~Point::DIM;
        }
    };

    KDTree(int maxSamples, const std::string& logFilename);
    ~KDTree(void);

    void InitialSampling(void);
    void Construct(void);
    void UpdateIndices(void);

    void ComputeErrors(void);
    void AdaptiveSampling(void);
    void Split(void);
    void PrepareLeafIndices(void);

    void Build(void);
    void SamplingPass(void);
    void Integrate(float4* pixels, uchar4* pixelsBytes, int width, int height);
    void SamplingDensity(float4* pixels, int width, int height);
    
    bool Validate(void);

    Buffer<Point>& GetSampleCoordinates(void) { return sampleCoordinates; }
    Buffer<float3>& GetSampleValues(void) { return sampleValues; }

    int GetCandidatesNum(void) { return candidatesNum; }
    int GetMaxLeafSize(void) { return maxLeafSize; }
    int GetNewSamples(void) { return newSamples; }
    int GetNumberOfSamples(void) { return numberOfSamples; }
    int GetNumberOfNodes(void) { return numberOfNodes; }
    int GetNumberOfLeaves(void) { return (numberOfNodes >> 1) + 1; }
    int GetMaxSamples(void) { return maxSamples;  }

    float GetScaleX(void) { return scaleX; }
    float GetScaleY(void) { return scaleY; }

private:

    std::ofstream log;
    bool logStats;

    float initialSamplingTime;
    float constructTime;
    float updateIndicesTime;
    float computeErrorsTime;
    float adaptiveSamplingTime;
    float splitTime;
    float prepareLeafIndicesTime;
    float integrateTime;

    int candidatesNum;
    int maxLeafSize;
    int bitsPerDim;
    int extraImgBits;
    int newSamples;
    int numberOfSamples;
    int numberOfNodes;
    int maxSamples;
    bool swapBuffers;

    float scaleX;
    float scaleY;
    float errorThreshold;

    Buffer<unsigned int> seeds;
    Buffer<Point> sampleCoordinates;
    Buffer<float3> sampleValues;

    Buffer<Node> nodes;
    Buffer<float4> nodesxy;
    Buffer<float4> nodeszw;
    Buffer<unsigned long long> nodeLocks;

    Buffer<Point> leafSamples;
    Buffer<int> leafIndices[2];
    Buffer<int> outNodeIndices;
    Buffer<float> errors;

};

}  // namespace mdas

#endif /* _KDTREE_H_ */
