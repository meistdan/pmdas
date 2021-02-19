//
// Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//

#pragma once

#include <sutil/vec_math.h>

#include "ao.h"

namespace ao {

    extern "C" {
        __constant__ ao::LaunchParams params;
    }


    //------------------------------------------------------------------------------
    //
    //
    //
    //------------------------------------------------------------------------------


    struct Onb
    {
        __forceinline__ __device__ Onb(const float3& normal)
        {
            m_normal = normal;

            if (fabs(m_normal.x) > fabs(m_normal.z))
            {
                m_binormal.x = -m_normal.y;
                m_binormal.y = m_normal.x;
                m_binormal.z = 0;
            }
            else
            {
                m_binormal.x = 0;
                m_binormal.y = -m_normal.z;
                m_binormal.z = m_normal.y;
            }

            m_binormal = normalize(m_binormal);
            m_tangent = cross(m_binormal, m_normal);
        }

        __forceinline__ __device__ void inverse_transform(float3& p) const
        {
            p = p.x * m_tangent + p.y * m_binormal + p.z * m_normal;
        }

        float3 m_tangent;
        float3 m_binormal;
        float3 m_normal;
    };


    static __forceinline__ __device__ void cosine_sample_hemisphere(const float u1, const float u2, float3& p)
    {
        // Uniformly sample disk.
        const float r = sqrtf(u1);
        const float phi = 2.0f * M_PIf * u2;
        p.x = r * cosf(phi);
        p.y = r * sinf(phi);

        // Project up to hemisphere.
        p.z = sqrtf(fmaxf(0.0f, 1.0f - p.x * p.x - p.y * p.y));
    }


    //------------------------------------------------------------------------------
    //
    //
    //
    //------------------------------------------------------------------------------


    static __forceinline__ __device__ void traceRadiance(
        OptixTraversableHandle      handle,
        float3                      ray_origin,
        float3                      ray_direction,
        float                       tmin,
        float                       tmax,
        ao::PayloadRadiance* payload
    )
    {
        unsigned int occluded = 0;
        unsigned int r0 = float_as_int(payload->r0);
        unsigned int r1 = float_as_int(payload->r1);
        optixTrace(
            handle,
            ray_origin, ray_direction,
            tmin,
            tmax,
            0.0f,                     // rayTime
            OptixVisibilityMask(1),
            OPTIX_RAY_FLAG_NONE,
            ao::RAY_TYPE_RADIANCE,        // SBT offset
            ao::RAY_TYPE_COUNT,           // SBT stride
            ao::RAY_TYPE_RADIANCE,        // missSBTIndex
            occluded, r0, r1);
        payload->occluded = occluded;
    }


    static __forceinline__ __device__ bool traceOcclusion(
        OptixTraversableHandle handle,
        float3                 ray_origin,
        float3                 ray_direction,
        float                  tmin,
        float                  tmax
    )
    {
        unsigned int occluded = 0u;
        optixTrace(
            handle,
            ray_origin,
            ray_direction,
            tmin,
            tmax,
            0.0f,                    // rayTime
            OptixVisibilityMask(1),
            OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT,
            ao::RAY_TYPE_OCCLUSION,      // SBT offset
            ao::RAY_TYPE_COUNT,          // SBT stride
            ao::RAY_TYPE_OCCLUSION,      // missSBTIndex
            occluded);
        return occluded;
    }

    __forceinline__ __device__ void setPayloadOcclusion(bool occluded)
    {
        optixSetPayload_0(static_cast<unsigned int>(occluded));
    }


    __forceinline__ __device__ float2  getPayloadSample()
    {
        float2 sample;
        sample.x = int_as_float(optixGetPayload_1());
        sample.y = int_as_float(optixGetPayload_2());
        return sample;
    }

} // namespace ao
