//
// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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
#include <optix.h>

#include <cuda/LocalGeometry.h>
#include <cuda/helpers.h>
#include <cuda/random.h>
#include <sutil/vec_math.h>

#include "ao_cuda.h"

#define M_PI       3.14159265358979323846   // pi
#define M_PI_2     1.57079632679489661923   // pi/2
#define M_PI_4     0.785398163397448309616  // pi/4

//------------------------------------------------------------------------------
//
//
//
//------------------------------------------------------------------------------

extern "C" __global__ void __raygen__pinhole()
{
    const uint3  launch_idx = optixGetLaunchIndex();
    const uint3  launch_dims = optixGetLaunchDimensions();
    const float3 eye = ao::params.eye;
    const float3 U = ao::params.U;
    const float3 V = ao::params.V;
    const float3 W = ao::params.W;
    const int    subframe_index = ao::params.subframe_index;

    //
    // Generate camera ray
    //
    unsigned int seed = tea<4>(launch_idx.y * launch_dims.x + launch_idx.x, subframe_index);

    float result = 0.0f;
    int j = ao::params.samples_per_launch;
    do
    {
        const float2 subpixel_jitter = make_float2(rnd(seed), rnd(seed));
        const float2 d = 2.0f * make_float2((static_cast<float>(launch_idx.x) + subpixel_jitter.x) / static_cast<float>(launch_dims.x),
            (static_cast<float>(launch_idx.y) + subpixel_jitter.y) / static_cast<float>(launch_dims.y)) - 1.0f;
        float3 ray_direction = normalize(d.x * U + d.y * V + W);
        float3 ray_origin = eye;

        //
        // Trace camera ray
        //
        ao::PayloadRadiance payload;
        payload.occluded = 0;
        payload.r0 = rnd(seed);
        payload.r1 = rnd(seed);

        ao::traceRadiance(ao::params.handle, ray_origin, ray_direction,
            0.01f,  // tmin       // TODO: smarter offset
            1e16f,  // tmax
            &payload);

        // Add result
        result += payload.occluded == 0 ? 1.0f : 0.0f;

    } while (--j);

    //
    // Update results
    //
    const unsigned int image_index = launch_idx.y * launch_dims.x + launch_idx.x;
    float3             accum_color = make_float3(result / static_cast<float>(ao::params.samples_per_launch));

    if (subframe_index > 0)
    {
        const float  a = 1.0f / static_cast<float>(subframe_index + 1);
        const float3 accum_color_prev = make_float3(ao::params.accum_buffer[image_index]);
        accum_color = lerp(accum_color_prev, accum_color, a);
    }
    ao::params.accum_buffer[image_index] = make_float4(accum_color, 1.0f);
    ao::params.frame_buffer[image_index] = make_color(accum_color);
}


extern "C" __global__ void __raygen__pinhole_mdas()
{
    const uint3  launch_idx = optixGetLaunchIndex();
    const uint3  launch_dims = optixGetLaunchDimensions();
    const float3 eye = ao::params.eye;
    const float3 U = ao::params.U;
    const float3 V = ao::params.V;
    const float3 W = ao::params.W;
    const int linear_index = ao::params.sample_offset + launch_idx.y * launch_dims.x + launch_idx.x;

    //
    // Generate camera ray
    //
    float4 sample = *reinterpret_cast<float4*>(&ao::params.sample_coordinates[ao::params.sample_dim * linear_index]);
    const float2 d = 2.0f * make_float2(sample.x / ao::params.scale.x,
        sample.y / ao::params.scale.y) - 1.0f;
    float3 ray_direction = normalize(d.x * U + d.y * V + W);
    float3 ray_origin = eye;

    //
    // Trace camera ray
    //
    ao::PayloadRadiance payload;
    payload.occluded = 0;
    payload.r0 = sample.z;
    payload.r1 = sample.w;

    ao::traceRadiance(ao::params.handle, ray_origin, ray_direction,
        0.01f,  // tmin       // TODO: smarter offset
        1e16f,  // tmax
        &payload);

    // Add result
    float result = payload.occluded == 0 ? 1.0f : 0.0f;


    //
    // Update results
    //
    ao::params.sample_values[linear_index] = make_float3(result);
}


extern "C" __global__ void __miss__constant_radiance()
{
    ao::setPayloadOcclusion(false);
}


extern "C" __global__ void __closesthit__occlusion()
{
    ao::setPayloadOcclusion(true);
}


extern "C" __global__ void __closesthit__radiance()
{
    const ao::HitGroupData* hit_group_data = reinterpret_cast<ao::HitGroupData*>(optixGetSbtDataPointer());
    const LocalGeometry geom = getLocalGeometry(hit_group_data->geometry_data);

    // Normal
    const float3 ray_dir = optixGetWorldRayDirection();
    const float3 N = faceforward(geom.N, -ray_dir, geom.N);

    // Sample cosine hemisphere
    const float2 r = ao::getPayloadSample();
    float3 R;
    ao::cosine_sample_hemisphere(r.x, r.y, R);
    ao::Onb onb(N);
    onb.inverse_transform(R);

    // Test occlusion
    const float tmin = 0.001f;
    const float tmax = ao::params.radius;
    const bool  occluded = ao::traceOcclusion(ao::params.handle, geom.P, R, tmin, tmax);
    ao::setPayloadOcclusion(occluded);
}
