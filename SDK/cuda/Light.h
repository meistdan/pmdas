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
#pragma once
#include <sutil/Preprocessor.h>

struct Light
{
    Light() {}

    enum class Type : int
    {
        POINT   = 0,
        AMBIENT = 1,
        DISTANT = 2,
        AREA    = 3
    };

    struct Point
    {
        float3   color      CONST_STATIC_INIT( { 1.0f, 1.0f, 1.0f } );
        float    intensity  CONST_STATIC_INIT( 1.0f                 );
        float3   position   CONST_STATIC_INIT( {}                   );
    };

    struct Distant
    {
        float3   color      CONST_STATIC_INIT({ 1.0f, 1.0f, 1.0f });
        float    intensity  CONST_STATIC_INIT(1.0f);
        float3   direction  CONST_STATIC_INIT({});
        float    radius     CONST_STATIC_INIT(1.0e+5f);
    };

    struct Ambient
    {
        float3   color      CONST_STATIC_INIT( {1.0f, 1.0f, 1.0f} );
    };

    struct Area
    {
        float3   color      CONST_STATIC_INIT({ 5.0f, 5.0f, 5.0f });
        float    intensity  CONST_STATIC_INIT(5.0f);
        float3   o          CONST_STATIC_INIT({});
        float3   u          CONST_STATIC_INIT({});
        float3   v          CONST_STATIC_INIT({});
    };

    Type  type;

    union
    {
        Point   point;
        Distant distant;
        Ambient ambient;
        Area    area;
    };
};
