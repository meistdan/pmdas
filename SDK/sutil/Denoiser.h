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

#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>

#include <sutil/CUDAOutputBuffer.h>
#include <sutil/Exception.h>

#include <cuda_runtime.h>

#include <cstdlib>
#include <iomanip>

namespace sutil
{
    class Denoiser
    {
    public:

        // Initialize the API and push all data to the GPU -- normaly done only once per session
        void init(OptixDeviceContext context, uint32_t width, uint32_t height);

        // Execute the denoiser. In interactive sessions, this would be done once per frame/subframe
        void exec(CUDAOutputBuffer<float4>& input_buffer, CUDAOutputBuffer<float4>& output_buffer);

        // Cleanup state, deallocate memory -- normally done only once per render session
        void cleanup();


    private:
        OptixDeviceContext    m_context = nullptr;
        OptixDenoiser         m_denoiser = nullptr;
        OptixDenoiserParams   m_params = {};

        CUdeviceptr           m_intensity = 0;
        CUdeviceptr           m_scratch = 0;
        uint32_t              m_scratch_size = 0;
        CUdeviceptr           m_state = 0;
        uint32_t              m_state_size = 0;

        OptixImage2D          m_inputs[3] = {};
        OptixImage2D          m_output;

        uint32_t              m_width = 0;
        uint32_t              m_height = 0;

    };



    void Denoiser::init(OptixDeviceContext context, uint32_t width, uint32_t height)
    {
        m_width = width;
        m_height = height;
        m_context = context;

        //
        // Create denoiser
        //
        {
            OptixDenoiserOptions options = {};
            options.inputKind = OPTIX_DENOISER_INPUT_RGB;
            OPTIX_CHECK(optixDenoiserCreate(m_context, &options, &m_denoiser));
            OPTIX_CHECK(optixDenoiserSetModel(
                m_denoiser,
                OPTIX_DENOISER_MODEL_KIND_HDR,
                nullptr, // data
                0        // size
            ));
        }


        //
        // Allocate device memory for denoiser
        //
        {
            OptixDenoiserSizes denoiser_sizes;
            OPTIX_CHECK(optixDenoiserComputeMemoryResources(
                m_denoiser,
                m_width,
                m_height,
                &denoiser_sizes
            ));

            // NOTE: if using tiled denoising, we would set scratch-size to 
            //       denoiser_sizes.withOverlapScratchSizeInBytes
            m_scratch_size = static_cast<uint32_t>(denoiser_sizes.withoutOverlapScratchSizeInBytes);

            CUDA_CHECK(cudaMalloc(
                reinterpret_cast<void**>(&m_intensity),
                sizeof(float)
            ));
            CUDA_CHECK(cudaMalloc(
                reinterpret_cast<void**>(&m_scratch),
                m_scratch_size
            ));

            CUDA_CHECK(cudaMalloc(
                reinterpret_cast<void**>(&m_state),
                denoiser_sizes.stateSizeInBytes
            ));
            m_state_size = static_cast<uint32_t>(denoiser_sizes.stateSizeInBytes);

            
        }

        //
        // Setup denoiser
        //
        {
            OPTIX_CHECK(optixDenoiserSetup(
                m_denoiser,
                0,  // CUDA stream
                m_width,
                m_height,
                m_state,
                m_state_size,
                m_scratch,
                m_scratch_size
            ));


            m_params.denoiseAlpha = 0;
            m_params.hdrIntensity = m_intensity;
            m_params.blendFactor = 0.0f;
        }
    }


    void Denoiser::exec(CUDAOutputBuffer<float4>& input_buffer, CUDAOutputBuffer<float4>& output_buffer)
    {
        SUTIL_ASSERT(m_width == input_buffer.width());
        SUTIL_ASSERT(m_height == input_buffer.height());
        SUTIL_ASSERT(m_width == output_buffer.width());
        SUTIL_ASSERT(m_height == output_buffer.height());

        m_inputs[0].data = reinterpret_cast<CUdeviceptr>(input_buffer.map());
        m_inputs[0].width = input_buffer.width();
        m_inputs[0].height = input_buffer.height();
        m_inputs[0].rowStrideInBytes = input_buffer.width() * sizeof(float4);
        m_inputs[0].pixelStrideInBytes = sizeof(float4);
        m_inputs[0].format = OPTIX_PIXEL_FORMAT_FLOAT4;

        m_inputs[1].data = 0;
        m_inputs[2].data = 0;

        m_output.data = reinterpret_cast<CUdeviceptr>(output_buffer.map());
        m_output.width = input_buffer.width();
        m_output.height = input_buffer.height();
        m_output.rowStrideInBytes = input_buffer.width() * sizeof(float4);
        m_output.pixelStrideInBytes = sizeof(float4);
        m_output.format = OPTIX_PIXEL_FORMAT_FLOAT4;

        OPTIX_CHECK(optixDenoiserComputeIntensity(
            m_denoiser,
            0, // CUDA stream
            m_inputs,
            m_intensity,
            m_scratch,
            m_scratch_size
        ));

        OPTIX_CHECK(optixDenoiserInvoke(
            m_denoiser,
            0, // CUDA stream
            &m_params,
            m_state,
            m_state_size,
            m_inputs,
            1, // num input channels
            0, // input offset X
            0, // input offset y
            &m_output,
            m_scratch,
            m_scratch_size
        ));

        CUDA_SYNC_CHECK();
    }


    void Denoiser::cleanup()
    {
        // Cleanup resources
        if (m_denoiser)
        {
            optixDenoiserDestroy(m_denoiser);
            CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_intensity)));
            CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_scratch)));
            CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_state)));
        }
    }

}