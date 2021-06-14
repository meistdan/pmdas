#ifndef _KDTREE_H_
#define _KDTREE_H_

#include "buffer.h"
#include "aabb.h"
#include <fstream>

namespace mdas {

template <typename Point>
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
        __host__ __device__ int Left(void) {
            return left < 0 ? ~left : left;
        }
        __host__ __device__ int Right(void) {
            return right < 0 ? ~right : right;
        }
    };

    KDTree(
        int maxSamples,
        int candidatesNum,
        int bitsPerDim,
        int extraImgBits,
        float errorThreshold,
        float scaleX,
        float scaleY,
        std::ofstream* log
    );

    void InitialSampling(void);

    void ComputeErrors(void);
    void AdaptiveSampling(void);

    void SamplingPass(void);
    void Integrate(float4* pixels, uchar4* pixelsBytes, int width, int height);
    void SamplingDensity(float4* pixels, int width, int height);
    
    bool Validate(void);

    Buffer<Point>& GetSampleCoordinates(void) { return sampleCoordinates; }
    Buffer<float3>& GetSampleValues(void) { return sampleValues; }

    int GetCandidatesNum(void) { return candidatesNum; }
    int GetMaxLeafSize(void) { return maxLeafSize; }
    int GetNewSamples(void) { return newSamples; }
    int GetNewNodes(void) { return newNodes; }
    int GetNumberOfSamples(void) { return numberOfSamples; }
    int GetNumberOfNodes(void) { return numberOfNodes; }
    int GetMaxSamples(void) { return maxSamples;  }

    float GetScaleX(void) { return scaleX; }
    float GetScaleY(void) { return scaleY; }

private:

    std::ofstream* log;
    bool logStats;

    int candidatesNum;
    int maxLeafSize;
    int bitsPerDim;
    int extraImgBits;
    int newSamples;
    int newNodes;
    int numberOfSamples;
    int numberOfNodes;
    int maxSamples;

    float scaleX;
    float scaleY;
    float errorThreshold;

    Buffer<Point> sampleCoordinates;
    Buffer<float3> sampleValues;

    Buffer<Node> nodes;
    Buffer<AABB<Point>> nodeBoxes;
    Buffer<float> nodeErrors;

    Buffer<unsigned int> seeds;

};

}  // namespace mdas

#endif /* _KDTREE_H_ */
