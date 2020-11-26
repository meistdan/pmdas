#ifndef _POINT_H_
#define _POINT_H_

namespace mdas {

template <int D>
struct PointN {

    static const int DIM = D;
    
    float data[DIM];
    
    __host__ __device__ PointN(void) {
        for (int i = 0; i < DIM; ++i) data[i] = 0.0;
    }

    __host__ __device__ PointN(float x) {
        for (int i = 0; i < DIM; ++i) data[i] = x;
    }

    __host__ __device__ float& operator[](int i) { return data[i]; }
    __host__ __device__ const float& operator[](int i) const { return data[i]; }

    __host__ __device__ static PointN Min(const PointN& point0, const PointN& point1) {
        PointN result;
        for (int i = 0; i < DIM; ++i)
            result.data[i] = fminf(point0.data[i], point1.data[i]);
        return result;
    }

    __host__ __device__ static PointN Max(const PointN& point0, const PointN& point1) {
        PointN result;
        for (int i = 0; i < DIM; ++i)
            result.data[i] = fmaxf(point0.data[i], point1.data[i]);
        return result;
    }
 
    __host__ __device__ PointN operator+(void) const {
        return *this;
    }
 
    __host__ __device__ PointN operator-(void) const {
        PointN result;
        for (int i = 0; i < DIM; ++i) 
            result.data[i] = -data[i];
        return result;
    }

    __host__ __device__ PointN operator+(const PointN& point) const {
        PointN result;
        for (int i = 0; i < DIM; ++i)
            result.data[i] = data[i] + point.data[i];
        return result;
    }

    __host__ __device__ PointN operator-(const PointN& point) const {
        PointN result;
        for (int i = 0; i < DIM; ++i) result.data[i] = data[i] - point.data[i];
        return result;
    }

    __host__ __device__ PointN operator*(const PointN& point) const {
        PointN result;
        for (int i = 0; i < DIM; ++i) 
            result.data[i] = data[i] * point.data[i];
        return result;
    }

    __host__ __device__ PointN operator/(const PointN& point) const {
        PointN result;
        for (int i = 0; i < DIM; ++i) 
            result.data[i] = data[i] / point.data[i];
        return result;
    }

    __host__ __device__ PointN& operator+=(const PointN& point) {
        for (int i = 0; i < DIM; ++i) 
            data[i] += point.data[i];
        return *this;
    }

    __host__ __device__ PointN& operator-=(const PointN& point) {
        for (int i = 0; i < DIM; ++i) 
            data[i] -= point.data[i];
        return *this;
    }

    __host__ __device__ PointN& operator*=(const PointN& point) {
        for (int i = 0; i < DIM; ++i) data[i] *= point.data[i];
        return *this;
    }

    __host__ __device__ PointN& operator/=(const PointN& point) {
        for (int i = 0; i < DIM; ++i) data[i] /= point.data[i];
        return *this;
    }

    __host__ __device__ PointN& operator+=(float alpha) {
        for (int i = 0; i < DIM; ++i) 
            data[i] += alpha;
        return *this;
    }

    __host__ __device__ PointN& operator-=(float alpha) {
        for (int i = 0; i < DIM; ++i) 
            data[i] -= alpha;
        return *this;
    }

    __host__ __device__ PointN& operator*=(float alpha) {
        for (int i = 0; i < DIM; ++i) 
            data[i] *= alpha;
        return *this;
    }

    __host__ __device__ PointN& operator/=(float alpha) {
        for (int i = 0; i < DIM; ++i) 
            data[i] /= alpha;
        return *this;
    }

    __host__ __device__ friend PointN operator*(float alpha, const PointN& point) {
        PointN result;
        for (int i = 0; i < DIM; ++i) 
            result.data[i] = alpha * point.data[i];
        return result;
    }

    __host__ __device__ friend PointN operator*(const PointN& point, float alpha) {
        return alpha * point;
    }

    __host__ __device__ friend PointN operator/(float alpha, const PointN& point) {
        PointN result;
        for (int i = 0; i < DIM; ++i) 
            result.data[i] = alpha / point.data[i];
        return result;
    }

    __host__ __device__ friend PointN operator/(const PointN& point, float alpha) {
        return (1 / alpha) * point;
    }

    __host__ __device__ friend PointN operator+(float alpha, const PointN& point) {
        PointN result;
        for (int i = 0; i < DIM; ++i) 
            result.data[i] = alpha + point.data[i];
        return result;
    }

    __host__ __device__ friend PointN operator+(const PointN& point, float alpha) {
        PointN result;
        for (int i = 0; i < DIM; ++i) 
            result.data[i] = alpha + point.data[i];
        return result;
    }

    __host__ __device__ friend PointN operator-(float alpha, const PointN& point) {
        PointN result;
        for (int i = 0; i < DIM; ++i) 
            result.data[i] = alpha - point.data[i];
        return result;
    }

    __host__ __device__ friend PointN operator-(const PointN& point, float alpha) {
        PointN result;
        for (int i = 0; i < DIM; ++i) 
            result.data[i] = point.data[i] - alpha;
        return result;
    }

    __host__ __device__ static float Norm(const PointN& point) {
        float result = 0;
        for (int i = 0; i < DIM; ++i) 
            result += point.data[i] * point.data[i];
        result = std::sqrt(result);
        return result;
    }

    __host__ __device__ static float Distance(const PointN& point0, const PointN& point1) {
        return Norm(point0 - point1);
    }

    __host__ __device__ static float NormSquared(const PointN& point) {
        float result = 0;
        for (int i = 0; i < DIM; ++i) result += point.data[i] * point.data[i];
        return result;
    }

    __host__ __device__ static float DistanceSquared(const PointN& point0, const PointN& point1) {
        return NormSquared(point0 - point1);
    }

};

typedef PointN<4> Point;

}  // namespace mdas

#endif /* _POINT_H_ */