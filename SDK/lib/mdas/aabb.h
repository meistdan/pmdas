#ifndef _AABB_H_
#define _AABB_H_

#include "point.h"
#include <float.h>

namespace mdas {

template <typename Point>
struct AABB {

    Point mn;
    Point mx;

    __host__ __device__ AABB(void) : mn(FLT_MAX), mx(-FLT_MAX) {}

    __host__ __device__ Point Center(void) { return 0.5 * (mx + mn); }
    __host__ __device__ Point Diagonal(void) { return mx - mn; }

    __host__ __device__ float Volume(void) {
        float res = 1.0;
        Point diag = Diagonal();
        for (int i = 0; i < Point::DIM; ++i) res *= diag[i];
        return res;
    }

    __host__ __device__ float SubVolume(void) {
        float res = 1.0;
        Point diag = Diagonal();
        for (int i = 0; i < Point::DIM - 2; ++i) res *= diag[i];
        return res;
    }

    __host__ __device__ int LargestAxis(void) {
        int maxAxis = -1;
        float maxExtent = -FLT_MAX;
        Point diag = Diagonal();
        for (int i = Point::DIM - 1; i >= 0; --i) {
            if (maxExtent < diag[i]) {
                maxExtent = diag[i];
                maxAxis = i;
            }
        }
        return maxAxis;
    }

    __host__ __device__ float MaxExtent(void) {
        float maxExtent = -FLT_MAX;
        Point diag = Diagonal();
        for (int i = Point::DIM - 1; i >= 0; --i) {
            if (maxExtent < diag[i])
                maxExtent = diag[i];
        }
        return maxExtent;
    }

    __host__ __device__ bool Contains(const Point& point) {
        for (int i = 0; i < Point::DIM; ++i) {
            if (point[i] < mn[i] || point[i] > mx[i]) return false;
        }
        return true;
    }

    __host__ __device__ void Union(const Point& point) {
        mn = Point::Min(mn, point);
        mx = Point::Max(mx, point);
    }

    __host__ __device__ void Union(const AABB& box) {
        mn = Point::Min(mn, box.mn);
        mx = Point::Max(mx, box.mx);
    }

    __host__ __device__ void Intersection(const AABB& box) {
        mn = Point::Max(mn, box.mn);
        mx = Point::Min(mx, box.mx);
    }

    __host__ __device__ bool Valid(void) {
        for (int i = 0; i < Point::DIM; ++i)
            if (mn[i] > mx[i]) return false;
        return true;
    }

    __host__ __device__ void Cubify(float R = 0.5f) {
        Point center = Center();
        float radius = R * Point::Distance(mn, mx);
        Point diag = Point(radius);
        mn = center - diag;
        mx = center + diag;
    }

    __host__ __device__ static bool Intersect(const AABB& a, const AABB& b) {
        AABB c = a;
        c.Intersection(b);
        return c.Valid();
    }

};

}  // namespace mdas

#endif /* _AABB_H_ */