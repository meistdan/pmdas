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
#include <sutil/Denoiser.h>
#include <sutil/Trackball.h>
#include <sutil/CUDAOutputBuffer.h>
#include <sutil/Environment.h>
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

int32_t           max_samples = 30000000;
float             samples_per_launch = 1;
bool              mdas_on = false;

whitted::LaunchParams* d_params = nullptr;
whitted::LaunchParams   params = {};
int32_t                 width = 1920;
int32_t                 height = 1080;

// MDAS
typedef mdas::Point4 Point;
mdas::KDTree<Point>* kdtree = nullptr;

// Denoiser
sutil::Denoiser denoiser;

//------------------------------------------------------------------------------
//
// Environment
//
//------------------------------------------------------------------------------

class AppEnvironment : public Environment {

protected:

    void registerOptions(void) {

        registerOption("Camera.position", OPT_VECTOR3);
        registerOption("Camera.direction", OPT_VECTOR3);
        registerOption("Camera.upVector", "0.0 1.0 0.0", OPT_VECTOR3);
        registerOption("Camera.fovy", "45.0", OPT_FLOAT);
        registerOption("Camera.focalDistance", "0.0", OPT_FLOAT);
        registerOption("Camera.lensRadius", "0.0", OPT_FLOAT);

        registerOption("Film.filename", OPT_STRING);
        registerOption("Film.width", "1024", OPT_INT);
        registerOption("Film.height", "768", OPT_INT);

        registerOption("Sampler.samples", "1", OPT_FLOAT);
        registerOption("Sampler.mdas", "0", OPT_BOOL);

        registerOption("Mdas.scaleFactor", "1.0", OPT_FLOAT);
        registerOption("Mdas.errorThreshold", "0.025", OPT_FLOAT);
        registerOption("Mdas.bitsPerDim", "0", OPT_INT);
        registerOption("Mdas.extraImgBits", "8", OPT_INT);
        registerOption("Mdas.candidatesNum", "4", OPT_INT);

        registerOption("Light.point", OPT_VECTOR3);
        registerOption("Light.distant", OPT_VECTOR3);
        registerOption("Light.color", OPT_VECTOR3);

        registerOption("Model.filename", OPT_STRING);
        registerOption("Model.frame", OPT_FRAME);
        registerOption("Model.frame", OPT_FRAME);
    }

public:
    
    AppEnvironment(void) {
        registerOptions();
    }

};

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
    else if (key == GLFW_KEY_W)
    {
        camera.setFocalDistance(camera.focalDistance() + 1.0f);
        camera_changed = true;
        std::cout << "Focal distance " << camera.focalDistance() << std::endl;
    }
    else if (key == GLFW_KEY_S)
    {
        camera.setFocalDistance(camera.focalDistance() - 1.0f);
        camera_changed = true;
        std::cout << "Focal distance " << camera.focalDistance() << std::endl;
    }
    else if (key == GLFW_KEY_A)
    {
        camera.setLensRadius(camera.lensRadius() - 0.001f);
        camera_changed = true;
        std::cout << "Lens radius " << camera.lensRadius() << std::endl;
    }
    else if (key == GLFW_KEY_D)
    {
        camera.setLensRadius(camera.lensRadius() + 0.001f);
        camera_changed = true;
        std::cout << "Lens radius " << camera.lensRadius() << std::endl;
    }
    else if (key == GLFW_KEY_C)
    {
        std::cout << "Eye " << camera.eye().x << " " << camera.eye().y << " " << camera.eye().z << std::endl;
        std::cout << "Direction " << camera.direction().x << " " << camera.direction().y << " " << camera.direction().z << std::endl;
        std::cout << "Up " << camera.up().x << " " << camera.up().y << " " << camera.up().z << std::endl;
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
    std::cerr << "Options: --no-gl-interop             Disable GL interop for display\n";
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

    // Parse lights
    std::vector<Light> lights;
    std::vector<float3> point_lights;
    std::vector<float3> distant_lights;
    std::vector<float3> light_colors;
    Environment::getInstance()->getVector3Values("Light.point", point_lights);
    Environment::getInstance()->getVector3Values("Light.distant", distant_lights);
    Environment::getInstance()->getVector3Values("Light.color", light_colors);
    
    // Ambient light
    Light light;
    light.type = Light::Type::AMBIENT;
    light.ambient.color = make_float3(0.35f, 0.35f, 0.35f);
    lights.push_back(light);

    // Point lights
    for (int i = 0; i < point_lights.size(); ++i) 
    {
        light.type = Light::Type::POINT;
        light.point.color = i < light_colors.size() ? light_colors[i] : make_float3(1.0f, 1.0f, 0.8f);
        light.point.intensity = 5.0f;
        light.point.position = point_lights[i];
        lights.push_back(light);
    }

    // Distant lights
    for (int i = 0; i < distant_lights.size(); ++i) 
    {
        light.type = Light::Type::DISTANT;
        light.distant.color = point_lights.size() + i < light_colors.size()
            ? light_colors[i] : make_float3(1.0f, 1.0f, 0.8f);
        light.distant.intensity = 1.0f;
        light.distant.direction = normalize(distant_lights[i]);
        light.distant.radius = length(scene.aabb().m_max - scene.aabb().m_min) * 0.5f;
        lights.push_back(light);
    }

    // No lights => use default lights
    if (point_lights.empty() && distant_lights.empty())
    {
        const float loffset = scene.aabb().maxExtent();
        light.type = Light::Type::POINT;
        light.point.color = { 1.0f, 1.0f, 0.8f };
        light.point.intensity = 5.0f;
        light.point.position = scene.aabb().center() + make_float3(loffset);
        lights.push_back(light);
        light.type = Light::Type::POINT;
        light.point.color = { 0.8f, 0.8f, 1.0f };
        light.point.intensity = 3.0f;
        light.point.position = scene.aabb().center() + make_float3(-loffset, 0.5f * loffset, -0.5f * loffset);
        lights.push_back(light);
    }

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

    params.samples_per_launch = std::max(static_cast<unsigned int>(samples_per_launch), 1u);
    params.miss_color = make_float3(0.1f);

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_params), sizeof(whitted::LaunchParams)));

    params.handle = scene.traversableHandle();
}


void initMdas(std::ofstream* log = nullptr)
{
    float errorThreshold, scaleFactor, scaleY;
    int bitsPerDim, extraImgBits, candidatesNum;

    Environment::getInstance()->getFloatValue("Mdas.errorThreshold", errorThreshold);
    Environment::getInstance()->getFloatValue("Mdas.scaleFactor", scaleFactor);
    Environment::getInstance()->getIntValue("Mdas.bitsPerDim", bitsPerDim);
    Environment::getInstance()->getIntValue("Mdas.extraImgBits", extraImgBits);
    Environment::getInstance()->getIntValue("Mdas.candidatesNum", candidatesNum);

    kdtree = new mdas::KDTree<Point>(
        max_samples, 
        candidatesNum,
        bitsPerDim,
        extraImgBits,
        errorThreshold,
        scaleFactor * width,
        scaleFactor * height,
        log
        );
    kdtree->InitialSampling();

    params.scale = make_float2(kdtree->GetScaleX(), kdtree->GetScaleY());
    params.sample_coordinates = reinterpret_cast<float*>(kdtree->GetSampleCoordinates().Data());
    params.sample_values = kdtree->GetSampleValues().Data();
    params.sample_count = kdtree->GetNumberOfSamples();
    params.sample_offset = 0;
    params.sample_dim = Point::DIM;
}


void initDenoising(const sutil::Scene& scene)
{
    denoiser.init(scene.context(), width, height);
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
}


void handleResize(
    sutil::CUDAOutputBuffer<float4>& output_buffer,
    sutil::CUDAOutputBuffer<uchar4>& output_buffer_bytes)
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
        if (mdas_on)
        {
            kdtree->InitialSampling();
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
    if (!mdas_on)
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

    if (!mdas_on)
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
    kdtree->Validate();
    kdtree->Integrate(result_buffer_data, result_buffer_data_bytes, width, height);
    output_buffer.unmap();
    output_buffer_bytes.unmap();
}


void samplingDensityMdas(sutil::ImageBuffer& density_buffer)
{
    kdtree->SamplingDensity(reinterpret_cast<float4*>(density_buffer.data), width, height);
}


void execDenoising(sutil::CUDAOutputBuffer<float4>& input_buffer, sutil::CUDAOutputBuffer<float4>& output_buffer)
{
    denoiser.exec(input_buffer, output_buffer);
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

    float focalDistance, lensRadius, fovy;
    float3 position, direction, upVector;
    if (Environment::getInstance()->getVector3Value("Camera.position", position)) 
        camera.setEye(position);
    if (Environment::getInstance()->getVector3Value("Camera.direction", direction)) 
        camera.setDirection(direction);
    if (Environment::getInstance()->getVector3Value("Camera.upVector", upVector))
        camera.setUp(upVector);
    if (Environment::getInstance()->getFloatValue("Camera.fovy", fovy))
        camera.setFovY(fovy);
    if (Environment::getInstance()->getFloatValue("Camera.focalDistance", focalDistance))
        camera.setFocalDistance(focalDistance);
    if (Environment::getInstance()->getFloatValue("Camera.lensRadius", lensRadius)) 
        camera.setLensRadius(lensRadius);

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
    denoiser.cleanup();
    Environment::deleteInstance();
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
    std::vector<std::string> infiles;
    std::vector<Frame> frames;

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
    }

    Environment* env = new AppEnvironment();
    Environment::setInstance(env);
    if (!env->parse(argc, argv))
    {
        std::cerr << "Parsing failed environment file!" << std::endl;
        exit(EXIT_FAILURE);
    }

    Environment::getInstance()->getStringValue("Film.filename", outfile);
    Environment::getInstance()->getIntValue("Film.width", width);
    Environment::getInstance()->getIntValue("Film.height", height);

    Environment::getInstance()->getBoolValue("Sampler.mdas", mdas_on);
    Environment::getInstance()->getFloatValue("Sampler.samples", samples_per_launch);

    Environment::getInstance()->getStringValues("Model.filename", infiles);
    Environment::getInstance()->getFrameValues("Model.frame", frames);

    if (infiles.empty())
    {
        std::cerr << "Input GLTF file required!" << std::endl;
        printUsageAndExit(argv[0]);
    }

    size_t begins = 0;
    size_t ends = 0;
    for (auto& frame : frames) 
    {
        if (frame.time == 0.0f) ++begins;
        if (frame.time == 1.0f) ++ends;
        if (frame.time < 0.0f || frame.time > 1.0f) {
            std::cerr << "Frame times must be in [0,1]!" << std::endl;
            exit(EXIT_FAILURE);
        }
    }
    if (begins != ends)
    {
        std::cerr << "Unpaired frames!" << std::endl;
        exit(EXIT_FAILURE);
    }

    try
    {
        sutil::Denoiser denoiser;
        sutil::Scene::SamplingType sampling_type = mdas_on ?
            sutil::Scene::SAMPLING_TYPE_MDAS_DEPTH_OF_FIELD : sutil::Scene::SAMPLING_TYPE_RANDOM;
        sutil::Scene::TraceType trace_type = sutil::Scene::TRACE_TYPE_WHITTED;
        sutil::Scene scene(sampling_type, trace_type);

        const size_t frame_num = 11;
        std::vector<Frame> frameset_resampled(frame_num);
        for (size_t i = 0; i < frame_num; ++i)
            frameset_resampled[i].time = i * (1.0f / static_cast<float>(frame_num - 1));

        size_t j = 0;

        for (size_t i = 0; i < infiles.size(); ++i)
        {
            size_t mesh_offset = scene.meshes().size();
            sutil::loadScene(infiles[i].c_str(), scene);
            if (i >= infiles.size() - begins)
            {
                std::vector<Frame> frameset;
                while (frames[j].time != 1.0f)
                    frameset.push_back(frames[j++]);
                frameset.push_back(frames[j++]);
                for (auto& frame : frameset_resampled)
                {
                    size_t minK = 0;
                    float minDiff = FLT_MAX;
                    for (size_t k = 0; k < frameset.size(); ++k)
                    {
                        float diff = fabs(frame.time - frameset[k].time);
                        if (minDiff > diff)
                        {
                            minDiff = diff;
                            minK = k;
                        }
                    }
                    minK = 0;
                    frame.rotate = frameset[minK].rotate;
                    frame.translate = frameset[minK].translate;
                    frame.scale = frameset[minK].scale;
                }
                for (size_t k = mesh_offset; k < scene.meshes().size(); ++k)
                {
                    for (auto& frame : frameset)
                        scene.meshes()[k]->frames.push_back(frame);
                }
            }
        }
        scene.finalize();

        OPTIX_CHECK(optixInit()); // Need to initialize function table
        initCameraState(scene);
        initLaunchParams(scene);
        
        if (outfile.empty())
        {
            GLFWwindow* window = sutil::initUI("optixDepthOfField", width, height);
            glfwSetMouseButtonCallback(window, mouseButtonCallback);
            glfwSetCursorPosCallback(window, cursorPosCallback);
            glfwSetWindowSizeCallback(window, windowSizeCallback);
            glfwSetWindowIconifyCallback(window, windowIconifyCallback);
            glfwSetKeyCallback(window, keyCallback);
            glfwSetScrollCallback(window, scrollCallback);
            glfwSetWindowUserPointer(window, &params);

            if (mdas_on) initMdas();

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
                    if (mdas_on)
                    {
                        if (params.subframe_index == 0)
                            launchSubframe(output_buffer, output_buffer_bytes, scene);
                        if (params.subframe_index >= 1 && kdtree->GetNumberOfSamples() 
                            + kdtree->GetNumberOfLeaves() <= max_samples)
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

            std::ofstream log(outfile + ".log");
            log << "WIDTH\n" << width << std::endl;
            log << "HEIGHT\n" << height << std::endl;

            if (mdas_on) initMdas(&log);
            
            auto start = std::chrono::steady_clock::now();
            launchSubframe(output_buffer, output_buffer_bytes, scene);
            auto stop = std::chrono::steady_clock::now();
            std::chrono::duration<double, std::milli> time = stop - start;
            log << "TRACE TIME\n" << time.count() << std::endl;

            if (mdas_on)
            {
                int total_samples = static_cast<int>(samples_per_launch * width * height);
                std::cout << "Total samples " << total_samples << std::endl;
                std::cout << "Sampling..." << std::endl;
                while (kdtree->GetNumberOfSamples() < std::min(total_samples, max_samples))
                {
                    samplingPassMdas();
                    start = std::chrono::steady_clock::now();
                    launchSubframe(output_buffer, output_buffer_bytes, scene);
                    stop = std::chrono::steady_clock::now();
                    time = stop - start;
                    log << "TRACE TIME\n" << time.count() << std::endl;
                    std::cout << "Leaves " << kdtree->GetNumberOfLeaves() << ", New samples " << kdtree->GetNewSamples() <<
                        ", Samples " << kdtree->GetNumberOfSamples() << ", Nodes " << kdtree->GetNumberOfNodes() << std::endl;
                }
                std::cout << "Integrating..." << std::endl;
                integrateMdas(output_buffer, output_buffer_bytes);

                // Sampling density
                std::cout << "Exporting sampling density..." << std::endl;
                sutil::CUDAOutputBuffer<float4> output_density_buffer(output_buffer_type, width, height);
                std::string density_outfile = outfile.substr(0, outfile.length() - 4) + "-density.exr";
                sutil::ImageBuffer density_buffer;
                density_buffer.width = output_density_buffer.width();
                density_buffer.height = output_density_buffer.height();
                density_buffer.data = output_density_buffer.getHostPointer();
                density_buffer.pixel_format = sutil::BufferImageFormat::FLOAT4;
                samplingDensityMdas(density_buffer);
                sutil::saveImage(density_outfile.c_str(), density_buffer, true);
            }

            // Denoising
            std::cout << "Denoising..." << std::endl;
            initDenoising(scene);
            sutil::CUDAOutputBuffer<float4> output_denoised_buffer(output_buffer_type, width, height);
            start = std::chrono::steady_clock::now();
            execDenoising(output_buffer, output_denoised_buffer);
            stop = std::chrono::steady_clock::now();
            time = stop - start;
            log << "DENOISING TIME\n" << time.count() << std::endl;
            std::string denoised_outfile = outfile.substr(0, outfile.length() - 4) + "-denoised.exr";
            sutil::ImageBuffer denoised_buffer;
            denoised_buffer.width = output_denoised_buffer.width();
            denoised_buffer.height = output_denoised_buffer.height();
            denoised_buffer.data = output_denoised_buffer.getHostPointer();
            denoised_buffer.pixel_format = sutil::BufferImageFormat::FLOAT4;
            sutil::saveImage(denoised_outfile.c_str(), denoised_buffer, true);
            std::cout << "Done" << std::endl;

            log.close();

            sutil::ImageBuffer buffer;
            buffer.width = output_buffer.width();
            buffer.height = output_buffer.height();
            buffer.data = output_buffer_bytes.getHostPointer();
            buffer.pixel_format = sutil::BufferImageFormat::UNSIGNED_BYTE4;

            const std::string ext = outfile.substr(outfile.length() - 3);
            if (ext == "EXR" || ext == "exr")
            {
                buffer.data = output_buffer.getHostPointer();
                buffer.pixel_format = sutil::BufferImageFormat::FLOAT4;
            }
            sutil::saveImage(outfile.c_str(), buffer, true);

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
