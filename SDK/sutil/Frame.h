#pragma once

#include <sutil/sutilapi.h>
#include <sutil/vec_math.h>

struct Frame {
    float time;
    float3 scale;
    float3 translate;
    float4 rotate;
};
