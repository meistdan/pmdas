import os
import cv2
import numpy as np

scene_name = "Bistro"

home_drive = "C:/Users/meist/projects"
optix_exe = home_drive + "/optix/SDK/build/bin/Release/optixDepthOfField.exe"
base_dir = home_drive + "/optix/SDK/data/" + scene_name + "/"

# output dir
out_dir = os.path.join(base_dir, "mdas")
if not (os.path.exists(out_dir)):
    os.mkdir(out_dir)
os.chdir(base_dir)

testing_passes = 5
ref_enabled = False
ref_spp = 1024
spp = [0.25, 0.5, 1, 2, 4]


def run(spp, testing_pass, mdas, ref):
    # test name
    test_name = scene_name
    if ref:
        test_name += "-reference"
    else:
        if mdas:
            test_name += "-mdas"
        test_name += "-spp-" + str(spp)
        test_name += "-pass-" + str(testing_pass)

    # scene file
    scene_dir = os.path.join(base_dir, "Exterior")
    scene_filename = "exterior.gltf"
    scene_filename_full = os.path.join(scene_dir, scene_filename)

    # image filename
    image_filename = os.path.join(out_dir, test_name + ".exr")

    # execute
    print(test_name)
    if mdas:
        os.system(optix_exe + " --model \"" + scene_filename_full + "\" --file \"" + image_filename + "\"" + " -s " + str(spp) + " --mdas 1")
    else:
        os.system(optix_exe + " --model \"" + scene_filename_full + "\" --file \"" + image_filename + "\"" + " -s " + str(spp))


# reference
if ref_enabled:
    run(ref_spp, 0, False, True)

for t in range(testing_passes):
    for s in spp:
        run(s, t, True, False)
        if s >= 1:
            run(s, t, False, False)
