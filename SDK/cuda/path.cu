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

#include "path_cuda.h"

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
    const float3 eye = path::params.eye;
    const float3 U = path::params.U;
    const float3 V = path::params.V;
    const float3 W = path::params.W;
    const int    subframe_index = path::params.subframe_index;

    //
    // Generate camera ray
    //
    unsigned int seed = tea<4>(launch_idx.y * launch_dims.x + launch_idx.x, subframe_index);

    float3 result = make_float3(0.0f);
    int j = path::params.samples_per_launch;
    do
    {
        const float2 subpixel_jitter = make_float2(rnd(seed), rnd(seed));
        const float2 d =
            2.0f
            * make_float2((static_cast<float>(launch_idx.x) + subpixel_jitter.x) / static_cast<float>(launch_dims.x),
                (static_cast<float>(launch_idx.y) + subpixel_jitter.y) / static_cast<float>(launch_dims.y))
            - 1.0f;
        float3 ray_direction = normalize(make_float3(d.x, d.y, 1.0f));
        float3 ray_origin = eye;
        ray_direction = normalize(ray_direction.x * U + ray_direction.y * V + ray_direction.z * W);

        //
        // Trace camera ray
        //
        path::PayloadRadiance payload;
        payload.emitted = make_float3(0.f);
        payload.radiance = make_float3(0.f);
        payload.attenuation = make_float3(1.f);
        payload.countEmitted = true;
        payload.done = false;

        int depth = 0;
        for (;;)
        {
            payload.r0 = rnd(seed);
            payload.r1 = rnd(seed);

            traceRadiance(path::params.handle, ray_origin, ray_direction,
                0.01f,  // tmin       // TODO: smarter offset
                1e16f,  // tmax
                &payload);

            result += payload.emitted;
            result += payload.radiance * payload.attenuation;

            if (payload.done || depth >= path::params.max_depth)
                break;

            ray_origin = payload.origin;
            ray_direction = payload.direction;

            ++depth;
        }

    } while (--j);

    //
    // Update results
    //
    const unsigned int image_index = launch_idx.y * launch_dims.x + launch_idx.x;
    float3             accum_color = result / static_cast<float>(path::params.samples_per_launch);

    if (subframe_index > 0)
    {
        const float  a = 1.0f / static_cast<float>(subframe_index + 1);
        const float3 accum_color_prev = make_float3(path::params.accum_buffer[image_index]);
        accum_color = lerp(accum_color_prev, accum_color, a);
    }
    path::params.accum_buffer[image_index] = make_float4(accum_color, 1.0f);
    path::params.frame_buffer[image_index] = make_color(accum_color);
}


extern "C" __global__ void __raygen__pinhole_mdas()
{
    const uint3  launch_idx = optixGetLaunchIndex();
    const uint3  launch_dims = optixGetLaunchDimensions();
    const float3 eye = path::params.eye;
    const float3 U = path::params.U;
    const float3 V = path::params.V;
    const float3 W = path::params.W;
    const int linear_index = path::params.sample_offset + launch_idx.y * launch_dims.x + launch_idx.x;
    
    //
    // Generate camera ray
    //
    float2* sample_ptr = reinterpret_cast<float2*>(&path::params.sample_coordinates[path::params.sample_dim * linear_index]);
    float2 sample = *sample_ptr;
    const float2 d = 2.0f * make_float2(sample.x / path::params.scale.x,
        sample.y / path::params.scale.y) - 1.0f;
    float3 ray_direction = normalize(make_float3(d.x, d.y, 1.0f));
    float3 ray_origin = eye;
    ray_direction = normalize(ray_direction.x * U + ray_direction.y * V + ray_direction.z * W);

    //
    // Trace camera ray
    //
    path::PayloadRadiance payload;
    payload.emitted = make_float3(0.f);
    payload.radiance = make_float3(0.f);
    payload.attenuation = make_float3(1.f);
    payload.countEmitted = true;
    payload.done = false;

    float3 result = make_float3(0.0f);
    int depth = 0;
    for (;;)
    {
        sample = *(++sample_ptr);
        payload.r0 = sample.x;
        payload.r1 = sample.y;

        traceRadiance(path::params.handle, ray_origin, ray_direction,
            0.01f,  // tmin       // TODO: smarter offset
            1e16f,  // tmax
            &payload);

        result += payload.emitted;
        result += payload.radiance * payload.attenuation;

        if (payload.done || depth >= path::params.max_depth)
            break;

        ray_origin = payload.origin;
        ray_direction = payload.direction;

        ++depth;
    }

    //
    // Update results
    //
    path::params.sample_values[linear_index] = result;
}


extern "C" __global__ void __miss__constant_radiance()
{
    path::PayloadRadiance* payload = path::getPayload();
    payload->radiance = path::params.miss_color;
    payload->done = true;
}


extern "C" __global__ void __closesthit__occlusion()
{
    path::setPayloadOcclusion( true );
}


extern "C" __global__ void __closesthit__radiance()
{
    const path::HitGroupData* hit_group_data = reinterpret_cast<path::HitGroupData*>( optixGetSbtDataPointer() );
    const LocalGeometry          geom           = getLocalGeometry( hit_group_data->geometry_data );

    //
    // Retrieve material data
    //
    float3 base_color = make_float3( hit_group_data->material_data.pbr.base_color );
    if( hit_group_data->material_data.pbr.base_color_tex )
        base_color *= path::linearize(
            make_float3( tex2D<float4>( hit_group_data->material_data.pbr.base_color_tex, geom.UV.x, geom.UV.y ) ) );

    float  metallic  = hit_group_data->material_data.pbr.metallic;
    float  roughness = hit_group_data->material_data.pbr.roughness;
    float4 mr_tex    = make_float4( 1.0f );
    if( hit_group_data->material_data.pbr.metallic_roughness_tex )
        // MR tex is (occlusion, roughness, metallic )
        mr_tex = tex2D<float4>( hit_group_data->material_data.pbr.metallic_roughness_tex, geom.UV.x, geom.UV.y );
    roughness *= mr_tex.y;
    metallic *= mr_tex.z;

    //
    // Convert to material params
    //
    const float  F0         = 0.04f;
    const float3 diff_color = base_color * ( 1.0f - F0 ) * ( 1.0f - metallic );
    const float3 spec_color = lerp( make_float3( F0 ), base_color, metallic );
    const float  alpha      = roughness * roughness;

    float3 N = geom.N;
    if (hit_group_data->material_data.pbr.normal_tex)
    {
        const float4 NN =
            2.0f * tex2D<float4>(hit_group_data->material_data.pbr.normal_tex, geom.UV.x, geom.UV.y) - make_float4(1.0f);
        N = normalize(NN.x * normalize(geom.dpdu) + NN.y * normalize(geom.dpdv) + NN.z * geom.N);
    }

    path::PayloadRadiance* payload = path::getPayload();
    payload->emitted = make_float3(0.0f);

    const float z1 = payload->r0;
    const float z2 = payload->r1;
    
    float3 w_in;
    path::cosine_sample_hemisphere(z1, z2, w_in);
    path::Onb onb(N);
    onb.inverse_transform(w_in);
    
    payload->direction = w_in;
    payload->origin = geom.P;
    payload->attenuation *= diff_color;
    payload->countEmitted = false;

    //
    // compute direct lighting
    //

    float3 result = make_float3( 0.0f );

    for( int i = 0; i < path::params.lights.count; ++i )
    {
        Light light = path::params.lights.at<Light>(i);
        if( light.type == Light::Type::POINT || light.type == Light::Type::DISTANT )
        {
            // TODO: optimize
            const float3 L = light.type == Light::Type::POINT ?
                                        normalize(light.point.position - geom.P) :
                                        normalize(light.distant.direction);
            const float  L_dist  = light.type == Light::Type::POINT ? 
                                        length( light.point.position - geom.P ) :
                                        2.0f * light.distant.radius;
            const float3 V       = -normalize( optixGetWorldRayDirection() );
            const float3 H       = normalize( L + V );
            const float  N_dot_L = dot( N, L );
            const float  N_dot_V = dot( N, V );
            const float  N_dot_H = dot( N, H );
            const float  V_dot_H = dot( V, H );

            if( N_dot_L > 0.0f && N_dot_V > 0.0f )
            {
                const float tmin     = 0.001f;           // TODO
                const float tmax     = L_dist - 0.001f;  // TODO
                const bool  occluded = path::traceOcclusion(path::params.handle, geom.P, L, tmin, tmax);
                if( !occluded )
                {
                    const float3 F     = path::schlick( spec_color, V_dot_H );
                    const float  G_vis = path::vis( N_dot_L, N_dot_V, alpha );
                    const float  D     = path::ggxNormal( N_dot_H, alpha );

                    const float3 diff = ( 1.0f - F ) * diff_color / M_PIf;
                    const float3 spec = F * G_vis * D;

                    result += light.point.color * light.point.intensity * N_dot_L * ( diff + spec );
                }
            }
        }
    }

    payload->radiance = result;
}
