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

#include "direct_cuda.h"

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
    const float3 eye = direct::params.eye;
    const float3 U = direct::params.U;
    const float3 V = direct::params.V;
    const float3 W = direct::params.W;
    const int    subframe_index = direct::params.subframe_index;

    //
    // Generate camera ray
    //
    unsigned int seed = tea<4>(launch_idx.y * launch_dims.x + launch_idx.x, subframe_index);

    float3 result = make_float3(0.0f);
    int j = direct::params.samples_per_launch;
    do
    {
        const float2 subpixel_jitter = make_float2(rnd(seed), rnd(seed));
        const float2 d =
            2.0f
            * make_float2((static_cast<float>(launch_idx.x) + subpixel_jitter.x) / static_cast<float>(launch_dims.x),
                (static_cast<float>(launch_idx.y) + subpixel_jitter.y) / static_cast<float>(launch_dims.y))
            - 1.0f;
        float3 ray_direction = normalize(d.x * U + d.y * V + W);
        float3 ray_origin = eye;

        //
        // Trace camera ray
        //
        direct::PayloadRadiance payload;
        payload.radiance = make_float3(0.0f);
        payload.r0 = rnd(seed);
        payload.r1 = rnd(seed);

        traceRadiance(direct::params.handle, ray_origin, ray_direction,
            0.01f,  // tmin       // TODO: smarter offset
            1e16f,  // tmax
            &payload);

        // Add result
        result += payload.radiance;

    } while (--j);

    //
    // Update results
    //
    const unsigned int image_index = launch_idx.y * launch_dims.x + launch_idx.x;
    float3             accum_color = result / static_cast<float>(direct::params.samples_per_launch);

    if (subframe_index > 0)
    {
        const float  a = 1.0f / static_cast<float>(subframe_index + 1);
        const float3 accum_color_prev = make_float3(direct::params.accum_buffer[image_index]);
        accum_color = lerp(accum_color_prev, accum_color, a);
    }
    direct::params.accum_buffer[image_index] = make_float4(accum_color, 1.0f);
    direct::params.frame_buffer[image_index] = make_color(accum_color);
}


extern "C" __global__ void __raygen__pinhole_mdas()
{
    const uint3  launch_idx = optixGetLaunchIndex();
    const uint3  launch_dims = optixGetLaunchDimensions();
    const float3 eye = direct::params.eye;
    const float3 U = direct::params.U;
    const float3 V = direct::params.V;
    const float3 W = direct::params.W;
    const int linear_index = direct::params.sample_offset + launch_idx.y * launch_dims.x + launch_idx.x;
    
    //
    // Generate camera ray
    //
    float4 sample = *reinterpret_cast<float4*>(&direct::params.sample_coordinates[direct::params.sample_dim * linear_index]);
    const float2 d = 2.0f * make_float2(sample.x / direct::params.scale.x,
        sample.y / direct::params.scale.y) - 1.0f;
    float3 ray_direction = normalize(d.x * U + d.y * V + W);
    float3 ray_origin = eye;

    //
    // Trace camera ray
    //
    direct::PayloadRadiance payload;
    payload.radiance = make_float3(0.0f);
    payload.r0 = sample.z;
    payload.r1 = sample.w;

    traceRadiance(direct::params.handle, ray_origin, ray_direction,
        0.01f,  // tmin       // TODO: smarter offset
        1e16f,  // tmax
        &payload);

    //
    // Update results
    //
    direct::params.sample_values[linear_index] = payload.radiance;
}


extern "C" __global__ void __miss__constant_radiance()
{
    direct::PayloadRadiance* payload = direct::getPayload();
    if (direct::params.environment_map != 0)
    {
        const float3 direction = normalize(optixGetWorldRayDirection());
#if 0
        float theta = atan2f(direction.x, direction.z);
        theta = theta < 0.0f ? theta + (2.0f * M_PI) : theta;
        float phi = acosf(direction.y);
        float u = 1.0f - (theta / (2.0f * M_PI));
        float v = phi / M_PI;
#else
        float d = sqrtf(direction.x * direction.x + direction.y * direction.y);
        float r = d > 0 ? 0.159154943f * acosf(direction.z) / d : 0.0f;
        float u = 0.5f + direction.x * r;
        float v = -(0.5f + direction.y * r);
#endif
        float4 c = tex2D<float4>(direct::params.environment_map, u, v);
        payload->radiance = make_float3(c);
    }
    else 
    {
        payload->radiance = direct::params.miss_color;
    }
}


extern "C" __global__ void __closesthit__occlusion()
{
    direct::setPayloadOcclusion( true );
}


extern "C" __global__ void __closesthit__radiance()
{
    const direct::HitGroupData* hit_group_data = reinterpret_cast<direct::HitGroupData*>( optixGetSbtDataPointer() );
    const LocalGeometry          geom           = getLocalGeometry( hit_group_data->geometry_data );

    //
    // Retrieve material data
    //
    float3 base_color = make_float3( hit_group_data->material_data.pbr.base_color );
    float opacity = hit_group_data->material_data.pbr.base_color.w;
    if (hit_group_data->material_data.pbr.base_color_tex)
    {
        float4 tmp = tex2D<float4>(hit_group_data->material_data.pbr.base_color_tex, geom.UV.x, geom.UV.y);
        base_color *= direct::linearize(make_float3(tmp));
        opacity = tmp.w;
    }

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
    const float3 V = -normalize(optixGetWorldRayDirection());
    if (hit_group_data->material_data.pbr.normal_tex)
    {
        const float4 NN =
            2.0f * tex2D<float4>(hit_group_data->material_data.pbr.normal_tex, geom.UV.x, geom.UV.y) - make_float4(1.0f);
        N = normalize(NN.x * normalize(geom.dpdu) + NN.y * normalize(geom.dpdv) + NN.z * geom.N);
    }
    N = faceforward(N, V, N);
    const float  N_dot_V = dot(N, V);

    //
    // emissive geometry hit
    //
    direct::PayloadRadiance* payload = direct::getPayload();
    payload->radiance = make_float3(0.0f);
    if (opacity < 0.0f)
    {
        opacity = -opacity;
        payload->radiance = base_color;
    }

    //
    // compute direct lighting
    //
    const float z1 = payload->r0;
    const float z2 = payload->r1;
    for (int i = 0; i < direct::params.lights.count; ++i)
    {
        Light light = direct::params.lights.at<Light>(i);
        if (light.type == Light::Type::POINT || light.type == Light::Type::DISTANT || light.type == Light::Type::AREA)
        {
            // TODO: optimize
            float3 L;
            float L_dist = -1.0f;
            if (light.type == Light::Type::DISTANT)
            {
                L = normalize(light.distant.direction);
                L_dist = 2.0f * light.distant.radius;
            }
            else if (light.type == Light::Type::AREA)
            {
                const float3 light_pos = light.area.o + light.area.u * z1 + light.area.v * z2;
                L = normalize(light_pos - geom.P);
                L_dist = length(light_pos - geom.P);
            }
            else // Point
            {
                L = normalize(light.point.position - geom.P);
                L_dist = length(light.point.position - geom.P);
            }

            const float3 H = normalize(L + V);
            const float  N_dot_L = dot(N, L);
            const float  N_dot_H = dot(N, H);
            const float  V_dot_H = dot(V, H);

            if (N_dot_L > 0.0f)
            {
                const float tmin = 0.001f;           // TODO
                const float tmax = L_dist - 0.001f;  // TODO
                const bool  occluded = direct::traceOcclusion(direct::params.handle, geom.P, L, tmin, tmax);
                if (!occluded)
                {
                    const float3 F = direct::schlick(spec_color, V_dot_H);
                    const float  G_vis = direct::vis(N_dot_L, N_dot_V, alpha);
                    const float  D = direct::ggxNormal(N_dot_H, alpha);

                    const float3 diff = (1.0f - F) * diff_color / M_PIf;
                    const float3 spec = F * G_vis * D;

                    payload->radiance += opacity * light.point.color * light.point.intensity * N_dot_L * (diff + spec);
                }
            }
        }
    }

}
