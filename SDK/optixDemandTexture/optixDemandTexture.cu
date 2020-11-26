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

#include "optixDemandTexture.h"

#include <DemandLoading/DemandTextureContext.h>
#include <DemandLoading/Tex2D.h>

#include <optixPaging/optixPaging.h>

#include <sutil/vec_math.h>

#include <cuda/helpers.h>
#include <cuda_runtime.h>

// Whether to use tex2DLod or tex2DGrad
//#define USE_TEX2DLOD 1

extern "C" {
__constant__ Params params;
}

//------------------------------------------------------------------------------
//
// Per ray data for closets hit program and functions to access it
//
//------------------------------------------------------------------------------

struct RayPayload
{
    // Return value
    float3 rgb;

    // Ray differential
    float3 origin_dx;
    float3 origin_dy;
    float3 direction_dx;
    float3 direction_dy;

    // padding
    int pad;
};


static __forceinline__ __device__ void* unpackPointer( unsigned int i0, unsigned int i1 )
{
    const unsigned long long uptr = static_cast<unsigned long long>( i0 ) << 32 | i1;
    void*          ptr  = reinterpret_cast<void*>( uptr );
    return ptr;
}


static __forceinline__ __device__ void packPointer( void* ptr, unsigned int& i0, unsigned int& i1 )
{
    const unsigned long long uptr = reinterpret_cast<unsigned long long>( ptr );
    i0                  = uptr >> 32;
    i1                  = uptr & 0x00000000ffffffff;
}


static __forceinline__ __device__ RayPayload* getPRD()
{
    const unsigned int u0 = optixGetPayload_0();
    const unsigned int u1 = optixGetPayload_1();
    return reinterpret_cast<RayPayload*>( unpackPointer( u0, u1 ) );
}


//------------------------------------------------------------------------------
//
// Determine the image pixel to render based on the sample index for multi-gpu
//
//------------------------------------------------------------------------------

static const int TILE_WIDTH  = 8;
static const int TILE_HEIGHT = 4;

static __forceinline__ __device__ uint2 getWorkIndex( int gpu_idx, int sample_idx, int width, int height, int num_gpus )
{
    const int tile_strip_width    = TILE_WIDTH * num_gpus;
    const int tile_strip_height   = TILE_HEIGHT;
    const int num_tile_strip_cols = width / tile_strip_width + ( width % tile_strip_width == 0 ? 0 : 1 );

    const int tile_strip_idx     = sample_idx / ( TILE_WIDTH * TILE_HEIGHT );
    const int tile_strip_y       = tile_strip_idx / num_tile_strip_cols;
    const int tile_strip_x       = tile_strip_idx - tile_strip_y * num_tile_strip_cols;
    const int tile_strip_x_start = tile_strip_x * tile_strip_width;
    const int tile_strip_y_start = tile_strip_y * tile_strip_height;

    const int tile_pixel_idx = sample_idx - ( tile_strip_idx * TILE_WIDTH * TILE_HEIGHT );
    const int tile_pixel_y   = tile_pixel_idx / TILE_WIDTH;
    const int tile_pixel_x   = tile_pixel_idx - tile_pixel_y * TILE_WIDTH;

    const int tile_offset_x = ( gpu_idx + tile_strip_y % num_gpus ) % num_gpus * TILE_WIDTH;

    const int pixel_y = tile_strip_y_start + tile_pixel_y;
    const int pixel_x = tile_strip_x_start + tile_pixel_x + tile_offset_x;
    return make_uint2( pixel_x, pixel_y );
}


//------------------------------------------------------------------------------
//
// Utility functions
//
//------------------------------------------------------------------------------

// trace a ray
static __forceinline__ __device__ void trace( OptixTraversableHandle handle, float3 ray_origin, float3 ray_direction, float tmin, float tmax, RayPayload* prd )
{
    unsigned int u0, u1;
    packPointer( prd, u0, u1 );
    optixTrace( handle, ray_origin, ray_direction, tmin, tmax,
                0.0f,  // rayTime
                OptixVisibilityMask( 1 ), OPTIX_RAY_FLAG_NONE,
                RAY_TYPE_RADIANCE,  // SBT offset
                RAY_TYPE_COUNT,     // SBT stride
                RAY_TYPE_RADIANCE,  // missSBTIndex
                u0, u1 );
}


// Convert Cartesian coordinates to polar coordinates
__forceinline__ __device__ float3 cartesian_to_polar( const float3& v )
{
    float azimuth;
    float elevation;
    float radius = length( v );

    float r = sqrtf( v.x * v.x + v.y * v.y );
    if( r > 0.0f )
    {
        azimuth   = atanf( v.y / v.x );
        elevation = atanf( v.z / r );

        if( v.x < 0.0f )
            azimuth += M_PIf;
        else if( v.y < 0.0f )
            azimuth += M_PIf * 2.0f;
    }
    else
    {
        azimuth = 0.0f;

        if( v.z > 0.0f )
            elevation = +M_PI_2f;
        else
            elevation = -M_PI_2f;
    }

    return make_float3( azimuth, elevation, radius );
}


// Compute texture derivatives in texture space from texture derivatives in world space
// and  ray differentials.
inline __device__ void computeTextureDerivatives( float2&       dpdx,  // texture derivative in x (out)
                                                  float2&       dpdy,  // texture derivative in y (out)
                                                  const float3& dPds,  // world space texture derivative
                                                  const float3& dPdt,  // world space texture derivative
                                                  float3        rdx,   // ray differential in x
                                                  float3        rdy,   // ray differential in y
                                                  const float3& normal,
                                                  const float3& rayDir )
{
    // Compute scale factor to project differentials onto surface plane
    float s = dot( rayDir, normal );

    // Clamp s to keep ray differentials from blowing up at grazing angles. Prevents overblurring.
    const float sclamp = 0.1f;
    if( s >= 0.0f && s < sclamp )
        s = sclamp;
    if( s < 0.0f && s > -sclamp )
        s = -sclamp;

    // Project the ray differentials to the surface plane.
    float tx = dot( rdx, normal ) / s;
    float ty = dot( rdy, normal ) / s;
    rdx -= tx * rayDir;
    rdy -= ty * rayDir;

    // Compute the texture derivatives in texture space. These are calculated as the
    // dot products of the projected ray differentials with the texture derivatives. 
    dpdx = make_float2( dot( dPds, rdx ), dot( dPdt, rdx ) );
    dpdy = make_float2( dot( dPds, rdy ), dot( dPdt, rdy ) );
}


//------------------------------------------------------------------------------
//
// Optix programs
//
//------------------------------------------------------------------------------

extern "C" __global__ void __raygen__rg()
{
    // Determine which pixel to render from the launch index
    const int imageWidth  = params.image_width;
    const int imageHeight = params.image_height;
    const uint3 launch_idx = optixGetLaunchIndex();
    unsigned int pixelIdx = launch_idx.x * imageWidth + launch_idx.y;
    const uint2 idx = getWorkIndex( params.device_idx, pixelIdx, imageWidth, imageHeight, params.num_devices );

    // Get the camera parameters
    const float3 U = params.U;
    const float3 V = params.V;
    const float3 W = params.W;
    const float2 d =
        2.0f * make_float2( static_cast<float>( idx.x ) / imageWidth, static_cast<float>( idx.y ) / imageHeight ) - 1.0f;

    // Construct the ray
    const float3 origin    = params.eye;
    const float3 direction = normalize( d.x * U + d.y * V + W );

    // Construct the ray payload with ray differentials
    RayPayload prd;
    prd.rgb          = make_float3( 0.0f );
    prd.origin_dx    = make_float3( 0.0f );
    prd.origin_dy    = make_float3( 0.0f );
    const float Wlen = length( W );
    // TODO: This is not 100% correct, since U and V are not perpendicular to the ray direction
    prd.direction_dx = U * ( 2.0f / ( imageWidth * Wlen ) );
    prd.direction_dy = V * ( 2.0f / ( imageHeight * Wlen ) );

    trace( params.handle, origin, direction,
           0.00f,  // tmin
           1e16f,  // tmax
           &prd );

    params.result_buffer[idx.y * params.image_width + idx.x] = make_color( prd.rgb );
}


extern "C" __global__ void __miss__ms()
{
    MissData*   rt_data = reinterpret_cast<MissData*>( optixGetSbtDataPointer() );
    RayPayload* prd     = getPRD();

    prd->rgb = make_float3( rt_data->r, rt_data->g, rt_data->b );
}


extern "C" __global__ void __intersection__is()
{
    HitGroupData* hg_data = reinterpret_cast<HitGroupData*>( optixGetSbtDataPointer() );
    const float3  orig    = optixGetObjectRayOrigin();
    const float3  dir     = optixGetObjectRayDirection();

    const float3 center = {0.f, 0.f, 0.f};
    const float  radius = hg_data->radius;
    const float3 O      = orig - center;
    const float  l      = 1 / length( dir );
    const float3 D      = dir * l;

    const float b    = dot( O, D );
    const float c    = dot( O, O ) - radius * radius;
    const float disc = b * b - c;
    if( disc > 0.0f )
    {
        const float sdisc = sqrtf( disc );
        const float root1 = ( -b - sdisc );

        const float  root11         = 0.0f;
        const float3 shading_normal = ( O + ( root1 + root11 ) * D ) / radius;

        float3 polar    = cartesian_to_polar( shading_normal );
        float3 texcoord = make_float3( polar.x * 0.5f * M_1_PIf, ( polar.y + M_PI_2f ) * M_1_PIf, polar.z / radius );

        unsigned int p0, p1, p2;
        p0 = float_as_int( texcoord.x );
        p1 = float_as_int( texcoord.y );
        p2 = float_as_int( texcoord.z );

        unsigned int n0, n1, n2;
        n0 = float_as_int( shading_normal.x );
        n1 = float_as_int( shading_normal.y );
        n2 = float_as_int( shading_normal.z );

        optixReportIntersection( root1,         // t hit
                                 0,             // user hit kind
                                 p0, p1, p2,    // texture coordinates
                                 n0, n1, n2 );  // geometric normal
    }
}


extern "C" __global__ void __closesthit__ch()
{
    // The demand-loaded texture id is provided in the hit group data.
    HitGroupData* hg_data      = reinterpret_cast<HitGroupData*>( optixGetSbtDataPointer() );
    unsigned int  textureId    = hg_data->demand_texture_id;
    const float   textureScale = hg_data->texture_scale;
    const float   radius       = hg_data->radius;

    // The texture coordinates and normal are calculated by the intersection shader are provided as attributes.
    const float3 texcoord = make_float3( int_as_float( optixGetAttribute_0() ), int_as_float( optixGetAttribute_1() ),
                                         int_as_float( optixGetAttribute_2() ) );

    const float3 N = make_float3( int_as_float( optixGetAttribute_3() ), int_as_float( optixGetAttribute_4() ),
                                  int_as_float( optixGetAttribute_5() ) );

    // Compute world space texture derivatives based on normal and radius, assuming a lat/long projection
    float3 dPds = radius * 2.0f * M_PIf * make_float3( N.y, -N.x, 0.0f );
    dPds /= dot( dPds, dPds );

    float3 dPdt = radius * M_PIf * normalize( cross( N, dPds ) );
    dPdt /= dot( dPdt, dPdt );

    // Compute final texture coordinates
    float s = texcoord.x * textureScale - 0.5f * (textureScale - 1.0f);
    float t = ( 1.0f - texcoord.y ) * textureScale - 0.5f * (textureScale - 1.0f);

    // Get the ray direction and hit distance
    RayPayload*  prd    = getPRD();
    const float3 rayDir = optixGetWorldRayDirection();
    const float  thit   = optixGetRayTmax();

    // Compute the ray differential values at the intersection point
    float3 rdx = prd->origin_dx + thit * prd->direction_dx;
    float3 rdy = prd->origin_dy + thit * prd->direction_dy;

    // Get texture space texture derivatives based on ray differentials
    float2 ddx, ddy;
    computeTextureDerivatives( ddx, ddy, dPds, dPdt, rdx, rdy, N, rayDir );

    // Scale the texture derivatives based on the texture scale (how many times the 
    // texture wraps around the sphere) and the mip bias
    float biasScale = exp2f( params.mipLevelBias );
    ddx *= textureScale * biasScale;
    ddy *= textureScale * biasScale;

    // Compute the color based on the render mode
    bool   isResident = true;
    float4 color      = make_float4( 0.0f );

    // Sample the texture a number of times
    const int        numTextureTaps = params.numTextureTaps;
    for( int i = 0; i < numTextureTaps; ++i )
    {
        color += tex2DGrad<float4>( params.demandTextureContext, textureId, s, t, ddx, ddy, &isResident );
        s += 0.0001f;
    }
    float invNumTextureTaps = (numTextureTaps != 0) ? numTextureTaps : 1;
    color *= invNumTextureTaps;
    prd->rgb = make_float3( color );
}
