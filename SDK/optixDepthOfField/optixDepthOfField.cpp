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

#include <glad/glad.h> // Needs to be included before gl_interop

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>

#include <sampleConfig.h>

#include <cuda/whitted.h>
#include <cuda/Light.h>

#include <sutil/Camera.h>
#include <sutil/Trackball.h>
#include <sutil/CUDAOutputBuffer.h>
#include <sutil/Exception.h>
#include <sutil/GLDisplay.h>
#include <sutil/Matrix.h>
#include <sutil/Scene.h>
#include <sutil/sutil.h>
#include <sutil/vec_math.h>

#include <GLFW/glfw3.h>

#include <array>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <random>

#include <lib/mdas/kdtree.h>

//#define USE_IAS // WAR for broken direct intersection of GAS on non-RTX cards

bool              resize_dirty = false;
bool              minimized = false;

// Camera state
bool              camera_changed = true;
sutil::Camera     camera;
sutil::Trackball  trackball;

// Mouse state
int32_t           mouse_button = -1;

int32_t           samples_per_launch = 16;
int32_t           number_of_lights = 16;
int32_t           mdas_iterations = -1;

whitted::LaunchParams* d_params = nullptr;
whitted::LaunchParams   params = {};
int32_t                 width = 1920;
int32_t                 height = 1080;

// MDAS
mdas::KDTree* kdtree = nullptr;

//------------------------------------------------------------------------------
//
// GLFW callbacks
//
//------------------------------------------------------------------------------

static void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods)
{
    double xpos, ypos;
    glfwGetCursorPos(window, &xpos, &ypos);

    if (action == GLFW_PRESS)
    {
        mouse_button = button;
        trackball.startTracking(static_cast<int>(xpos), static_cast<int>(ypos));
    }
    else
    {
        mouse_button = -1;
    }
}


static void cursorPosCallback(GLFWwindow* window, double xpos, double ypos)
{
    if (mouse_button == GLFW_MOUSE_BUTTON_LEFT)
    {
        trackball.setViewMode(sutil::Trackball::LookAtFixed);
        trackball.updateTracking(static_cast<int>(xpos), static_cast<int>(ypos), width, height);
        camera_changed = true;
    }
    else if (mouse_button == GLFW_MOUSE_BUTTON_RIGHT)
    {
        trackball.setViewMode(sutil::Trackball::EyeFixed);
        trackball.updateTracking(static_cast<int>(xpos), static_cast<int>(ypos), width, height);
        camera_changed = true;
    }
}


static void windowSizeCallback(GLFWwindow* window, int32_t res_x, int32_t res_y)
{
    // Keep rendering at the current resolution when the window is minimized.
    if (minimized)
        return;

    // Output dimensions must be at least 1 in both x and y.
    sutil::ensureMinimumSize(res_x, res_y);

    width = res_x;
    height = res_y;
    camera_changed = true;
    resize_dirty = true;
}


static void windowIconifyCallback(GLFWwindow* window, int32_t iconified)
{
    minimized = (iconified > 0);
}


static void keyCallback(GLFWwindow* window, int32_t key, int32_t /*scancode*/, int32_t action, int32_t /*mods*/)
{
    if (action == GLFW_PRESS)
    {
        if (key == GLFW_KEY_Q ||
            key == GLFW_KEY_ESCAPE)
        {
            glfwSetWindowShouldClose(window, true);
        }
    }
    else if (key == GLFW_KEY_C)
    {
        std::cout << "Camera: " << std::endl;
        std::cout << "\tEye " << camera.eye().x << " " << camera.eye().y << " " << camera.eye().z << std::endl;
        std::cout << "\tDirection " << camera.direction().x << " " << camera.direction().y << " " << camera.direction().z << std::endl;
    }
    else if (key == GLFW_KEY_W)
    {
        camera.setFocalDistance(camera.focalDistance() + 0.5f);
        camera_changed = true;
        std::cout << "Focal distance " << camera.focalDistance() << std::endl;
    }
    else if (key == GLFW_KEY_S)
    {
        camera.setFocalDistance(camera.focalDistance() - 0.5f);
        camera_changed = true;
        std::cout << "Focal distance " << camera.focalDistance() << std::endl;
    }
    else if (key == GLFW_KEY_A)
    {
        camera.setLensRadius(camera.lensRadius() - 0.002f);
        camera_changed = true;
        std::cout << "Lens radius " << camera.lensRadius() << std::endl;
    }
    else if (key == GLFW_KEY_D)
    {
        camera.setLensRadius(camera.lensRadius() + 0.002f);
        camera_changed = true;
        std::cout << "Lens radius " << camera.lensRadius() << std::endl;
    }
}


static void scrollCallback(GLFWwindow* window, double xscroll, double yscroll)
{
    if (trackball.wheelEvent((int)yscroll))
        camera_changed = true;
}


//------------------------------------------------------------------------------
//
// Helper functions
// TODO: some of these should move to sutil or optix util header
//
//------------------------------------------------------------------------------

void printUsageAndExit(const char* argv0)
{
    std::cerr << "Usage  : " << argv0 << " [options]\n";
    std::cerr << "Options: --file | -f <filename>      File for image output\n";
    std::cerr << "          --dim=<width>x<height>      Set image dimensions; defaults to 768x768\n";
    std::cerr << "         --launch-samples | -s       Number of samples per pixel per launch (default 16)\n";
    std::cerr << "         --no-gl-interop             Disable GL interop for display\n";
    std::cerr << "         --model <model.gltf>        Specify model to render (required)\n";
    std::cerr << "         --lights | -l               Number of lights\n";
    std::cerr << "         --mdas                      Specify MDAS iterations (negative values turn MDAS off)\n";
    std::cerr << "         --help | -h                 Print this usage message\n";
    exit(0);
}


void initLaunchParams(const sutil::Scene& scene) {
    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&params.accum_buffer),
        width * height * sizeof(float4)
    ));
    params.frame_buffer = nullptr; // Will be set when output buffer is mapped

    params.subframe_index = 0u;

    const float loffset = scene.aabb().maxExtent();

#if 0
    // TODO: add light support to sutil::Scene
    std::vector<Light> lights(2);
    lights[0].type = Light::Type::POINT;
    lights[0].point.color = { 1.0f, 1.0f, 0.8f };
    lights[0].point.intensity = 5.0f;
    lights[0].point.position = scene.aabb().center() + make_float3(loffset);
    lights[0].point.falloff = Light::Falloff::QUADRATIC;
    lights[1].type = Light::Type::POINT;
    lights[1].point.color = { 0.8f, 0.8f, 1.0f };
    lights[1].point.intensity = 3.0f;
    lights[1].point.position = scene.aabb().center() + make_float3(-loffset, 0.5f * loffset, -0.5f * loffset);
    lights[1].point.falloff = Light::Falloff::QUADRATIC;
#else
    std::vector<Light> lights(number_of_lights);
    std::mt19937 mt;
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    lights[0].type = Light::Type::AMBIENT;
    lights[0].ambient.color = make_float3(0.35f, 0.35f, 0.35f);
    for (int i = 1; i < number_of_lights; ++i) {
        float u1 = dist(mt);
        float u2 = dist(mt);
        const float r = sqrtf(u1);
        const float phi = 2.0f * M_PIf * u2;
        float3 p;
        p.x = r * cosf(phi);
        p.y = r * sinf(phi);
        p.z = sqrtf(fmaxf(0.0f, 1.0f - p.x * p.x - p.y * p.y));
        lights[i].type = Light::Type::POINT;
        lights[i].point.color = { 1.0f, 1.0f, 0.8f };
        lights[i].point.intensity = 50.0f / number_of_lights;
        lights[i].point.position = scene.aabb().center() + loffset * p;
        lights[i].point.falloff = Light::Falloff::QUADRATIC;
    }
#endif

    params.lights.count = static_cast<uint32_t>(lights.size());
    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&params.lights.data),
        lights.size() * sizeof(Light)
    ));
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(params.lights.data),
        lights.data(),
        lights.size() * sizeof(Light),
        cudaMemcpyHostToDevice
    ));

    params.samples_per_launch = samples_per_launch;
    params.miss_color = make_float3(0.1f);

    //CUDA_CHECK( cudaStreamCreate( &stream ) );
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_params), sizeof(whitted::LaunchParams)));

    params.handle = scene.traversableHandle();
}


void initMdas()
{
    const size_t maxSamples = 10000000;
    kdtree = new mdas::KDTree(maxSamples);
    kdtree->Build();

    params.scale = make_float2(kdtree->GetScaleX(), kdtree->GetScaleY());
    params.sample_coordinates = kdtree->GetSampleCoordinates().Data();
    params.sample_values = kdtree->GetSampleValues().Data();
    params.sample_count = kdtree->GetNumberOfSamples();
    params.sample_offset = 0;
}


void handleCameraUpdate(whitted::LaunchParams& params)
{
    if (!camera_changed)
        return;
    camera_changed = false;

    camera.setAspectRatio(static_cast<float>(width) / static_cast<float>(height));
    params.eye = camera.eye();
    camera.UVWFrame(params.U, params.V, params.W);

    params.focal_distance = camera.focalDistance();
    params.lens_radius = camera.lensRadius();
    /*
    std::cerr
        << "Updating camera:\n"
        << "\tU: " << params.U.x << ", " << params.U.y << ", " << params.U.z << std::endl
        << "\tV: " << params.V.x << ", " << params.V.y << ", " << params.V.z << std::endl
        << "\tW: " << params.W.x << ", " << params.W.y << ", " << params.W.z << std::endl;
        */

}


void handleResize(sutil::CUDAOutputBuffer<float4>& output_buffer, sutil::CUDAOutputBuffer<uchar4>& output_buffer_bytes)
{
    if (!resize_dirty)
        return;
    resize_dirty = false;

    output_buffer.resize(width, height);
    output_buffer_bytes.resize(width, height);
}


void updateState(
    sutil::CUDAOutputBuffer<float4>& output_buffer,
    sutil::CUDAOutputBuffer<uchar4>& output_buffer_bytes,
    whitted::LaunchParams& params
)
{
    // Update params on device
    if (camera_changed || resize_dirty)
    {
        params.subframe_index = 0;
        if (mdas_iterations >= 0)
        {
            kdtree->Build();
            params.sample_count = kdtree->GetNumberOfSamples();
            params.sample_offset = 0;
        }
    }

    handleCameraUpdate(params);
    handleResize(output_buffer, output_buffer_bytes);
}


void launchSubframe(
    sutil::CUDAOutputBuffer<float4>& output_buffer,
    sutil::CUDAOutputBuffer<uchar4>& output_buffer_bytes,
    const sutil::Scene& scene
)
{
    if (mdas_iterations < 0)
    {
        float4* result_buffer_data = output_buffer.map();
        uchar4* result_buffer_data_bytes = output_buffer_bytes.map();
        params.accum_buffer = result_buffer_data;
        params.frame_buffer = result_buffer_data_bytes;
    }

    CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(d_params),
        &params,
        sizeof(whitted::LaunchParams),
        cudaMemcpyHostToDevice,
        0 // stream
    ));

    if (mdas_iterations < 0)
    {
        OPTIX_CHECK(optixLaunch(
            scene.pipeline(),
            0,             // stream
            reinterpret_cast<CUdeviceptr>(d_params),
            sizeof(whitted::LaunchParams),
            scene.sbt(),
            width,  // launch width
            height, // launch height
            1       // launch depth
        ));
        output_buffer_bytes.unmap();
        output_buffer.unmap();
    }
    else
    {
        OPTIX_CHECK(optixLaunch(
            scene.pipeline(),
            0,             // stream
            reinterpret_cast<CUdeviceptr>(d_params),
            sizeof(whitted::LaunchParams),
            scene.sbt(),
            params.sample_count,  // launch width
            1,                    // launch height
            1                     // launch depth
        ));
    }
    CUDA_SYNC_CHECK();
}


void samplingPassMdas()
{
    kdtree->SamplingPass();
    kdtree->Validate();
    params.sample_count = kdtree->GetNewSamples();
    params.sample_offset = kdtree->GetNumberOfSamples() - kdtree->GetNewSamples();
}


void integrateMdas(sutil::CUDAOutputBuffer<float4>& output_buffer, sutil::CUDAOutputBuffer<uchar4>& output_buffer_bytes)
{
    float4* result_buffer_data = output_buffer.map();
    uchar4* result_buffer_data_bytes = output_buffer_bytes.map();
    kdtree->UpdateIndices();
    //kdtree->Validate();
    kdtree->Integrate(result_buffer_data, result_buffer_data_bytes, width, height);
    output_buffer.unmap();
    output_buffer_bytes.unmap();
}


void displaySubframe(
    sutil::CUDAOutputBuffer<uchar4>& output_buffer,
    sutil::GLDisplay& gl_display,
    GLFWwindow* window)
{
    // Display
    int framebuf_res_x = 0;   // The display's resolution (could be HDPI res)
    int framebuf_res_y = 0;   //
    glfwGetFramebufferSize(window, &framebuf_res_x, &framebuf_res_y);
    gl_display.display(
        output_buffer.width(),
        output_buffer.height(),
        framebuf_res_x,
        framebuf_res_y,
        output_buffer.getPBO()
    );
}


void initCameraState(const sutil::Scene& scene)
{
    camera = scene.camera();
    //camera.setFocalDistance(1.0f);
    //camera.setLensRadius(0.005f);
    //camera.setFocalDistance(20.0f);
    //camera.setLensRadius(0.3f);
    camera_changed = true;

    trackball.setCamera(&camera);
    trackball.setMoveSpeed(10.0f);
    trackball.setReferenceFrame(make_float3(1.0f, 0.0f, 0.0f), make_float3(0.0f, 0.0f, 1.0f), make_float3(0.0f, 1.0f, 0.0f));
    trackball.setGimbalLock(true);
}


void cleanup()
{
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(params.lights.data)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_params)));
    if (kdtree != nullptr) delete kdtree;
}


//------------------------------------------------------------------------------
//
// Main
//
//------------------------------------------------------------------------------

int main(int argc, char* argv[])
{
    sutil::CUDAOutputBufferType output_buffer_type = sutil::CUDAOutputBufferType::GL_INTEROP;

    //
    // Parse command line options
    //
    std::string outfile;
    std::string infile = sutil::sampleDataFilePath("WaterBottle/WaterBottle.gltf");

    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];
        if (arg == "--help" || arg == "-h")
        {
            printUsageAndExit(argv[0]);
        }
        else if (arg == "--no-gl-interop")
        {
            output_buffer_type = sutil::CUDAOutputBufferType::CUDA_DEVICE;
        }
        else if (arg == "--model")
        {
            if (i >= argc - 1)
                printUsageAndExit(argv[0]);
            infile = argv[++i];
        }
        else if (arg == "--file" || arg == "-f")
        {
            if (i >= argc - 1)
                printUsageAndExit(argv[0]);
            outfile = argv[++i];
        }
        else if (arg.substr(0, 6) == "--dim=")
        {
            const std::string dims_arg = arg.substr(6);
            sutil::parseDimensions(dims_arg.c_str(), width, height);
        }
        else if (arg == "--launch-samples" || arg == "-s")
        {
            if (i >= argc - 1)
                printUsageAndExit(argv[0]);
            samples_per_launch = atoi(argv[++i]);
        }
        else if (arg == "--lights" || arg == "-l")
        {
            if (i >= argc - 1)
                printUsageAndExit(argv[0]);
            number_of_lights = atoi(argv[++i]);
        }
        else if (arg == "--mdas")
        {
            if (i >= argc - 1)
                printUsageAndExit(argv[0]);
            mdas_iterations = atoi(argv[++i]);
        }
        else
        {
            std::cerr << "Unknown option '" << argv[i] << "'\n";
            printUsageAndExit(argv[0]);
        }
    }

    if (infile.empty())
    {
        std::cerr << "--model argument required" << std::endl;
        printUsageAndExit(argv[0]);
    }


    try
    {
        sutil::Scene scene(mdas_iterations >= 0);
        sutil::loadScene(infile.c_str(), scene);
        scene.finalize();

        OPTIX_CHECK(optixInit()); // Need to initialize function table
        initCameraState(scene);
        initLaunchParams(scene);
        if (mdas_iterations >= 0)
            initMdas();

        if (outfile.empty())
        {
            GLFWwindow* window = sutil::initUI("optixMeshViewer", width, height);
            glfwSetMouseButtonCallback(window, mouseButtonCallback);
            glfwSetCursorPosCallback(window, cursorPosCallback);
            glfwSetWindowSizeCallback(window, windowSizeCallback);
            glfwSetWindowIconifyCallback(window, windowIconifyCallback);
            glfwSetKeyCallback(window, keyCallback);
            glfwSetScrollCallback(window, scrollCallback);
            glfwSetWindowUserPointer(window, &params);

            //
            // Render loop
            //
            {
                sutil::CUDAOutputBuffer<uchar4> output_buffer_bytes(output_buffer_type, width, height);
                sutil::CUDAOutputBuffer<float4> output_buffer(output_buffer_type, width, height);
                sutil::GLDisplay gl_display;

                std::chrono::duration<double> state_update_time(0.0);
                std::chrono::duration<double> render_time(0.0);
                std::chrono::duration<double> display_time(0.0);

                do
                {
                    auto t0 = std::chrono::steady_clock::now();
                    glfwPollEvents();
                    updateState(output_buffer, output_buffer_bytes, params);
                    auto t1 = std::chrono::steady_clock::now();
                    state_update_time += t1 - t0;
                    t0 = t1;
                    if (mdas_iterations >= 0)
                    {
                        if (params.subframe_index == 0)
                            launchSubframe(output_buffer, output_buffer_bytes, scene);
                        if (params.subframe_index >= 1 && params.subframe_index <= mdas_iterations)
                        {
                            samplingPassMdas();
                            std::cout << "Leaves " << kdtree->GetNumberOfLeaves() << ", New samples " << kdtree->GetNewSamples() <<
                                ", Samples " << kdtree->GetNumberOfSamples() << ", Nodes " << kdtree->GetNumberOfNodes() << std::endl;
                            launchSubframe(output_buffer, output_buffer_bytes, scene);
                        }
                        integrateMdas(output_buffer, output_buffer_bytes);
                    }
                    else
                    {
                        launchSubframe(output_buffer, output_buffer_bytes, scene);
                    }
                    t1 = std::chrono::steady_clock::now();
                    render_time += t1 - t0;
                    t0 = t1;

                    displaySubframe(output_buffer_bytes, gl_display, window);
                    t1 = std::chrono::steady_clock::now();
                    display_time += t1 - t0;

                    sutil::displayStats(state_update_time, render_time, display_time);

                    glfwSwapBuffers(window);

                    ++params.subframe_index;
                } while (!glfwWindowShouldClose(window));
                CUDA_SYNC_CHECK();
            }

            sutil::cleanupUI(window);
        }
        else
        {
            output_buffer_type = sutil::CUDAOutputBufferType::CUDA_DEVICE;

            sutil::CUDAOutputBuffer<uchar4> output_buffer_bytes(output_buffer_type, width, height);
            sutil::CUDAOutputBuffer<float4> output_buffer(output_buffer_type, width, height);

            handleCameraUpdate(params);
            handleResize(output_buffer, output_buffer_bytes);
            launchSubframe(output_buffer, output_buffer_bytes, scene);
            if (mdas_iterations >= 0)
            {
                for (int i = 0; i < mdas_iterations; ++i)
                {
                    samplingPassMdas();
                    launchSubframe(output_buffer, output_buffer_bytes, scene);
                    std::cout << "Leaves " << kdtree->GetNumberOfLeaves() << ", New samples " << kdtree->GetNewSamples() <<
                        ", Samples " << kdtree->GetNumberOfSamples() << ", Nodes " << kdtree->GetNumberOfNodes() << std::endl;
                }
                integrateMdas(output_buffer, output_buffer_bytes);
            }
            sutil::ImageBuffer buffer;
            buffer.width = output_buffer.width();
            buffer.height = output_buffer.height();
            buffer.data = output_buffer_bytes.getHostPointer();
            buffer.pixel_format = sutil::BufferImageFormat::UNSIGNED_BYTE4;

            const std::string ext = outfile.substr(outfile.length() - 3);
            if (ext == "PPM" || ext == "ppm")
            {
                buffer.data = output_buffer.getHostPointer();
                buffer.pixel_format = sutil::BufferImageFormat::FLOAT4;
            }

            sutil::saveImage(outfile.c_str(), buffer, false);

            if (output_buffer_type == sutil::CUDAOutputBufferType::GL_INTEROP)
            {
                glfwTerminate();
            }
        }

        cleanup();

    }
    catch (std::exception& e)
    {
        std::cerr << "Caught exception: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
