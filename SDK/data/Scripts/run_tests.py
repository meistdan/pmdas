import os
import cv2
import numpy as np

scene_name = "Bistro"

home_drive = "C:/Users/meist/projects"
optix_exe = home_drive + "/optix/SDK/build/bin/Release/optixDepthOfField.exe"
base_dir = home_drive + "/optix/SDK/data/" + scene_name + "/"

# output dir
out_dir = os.path.join(base_dir, "mdas-8bit")
if not (os.path.exists(out_dir)):
    os.mkdir(out_dir)
os.chdir(base_dir)

ref_spp = 1024
#spp = [0.25, 0.5, 1, 2, 4]
spp = [4]


def read_image(filename):
    img = cv2.imread(filename)
    return img


def mse(img0: np.array, img1 : np.array):
    return (np.subtract(img0, img1) ** 2).mean()


def run(spp, mdas, ref):
    # test name
    test_name = scene_name
    if ref:
        test_name += "-reference"
    else:
        if mdas:
            test_name += "-mdas"
        test_name += "-spp-" + str(spp)

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
#run(ref_spp, False, True)

for s in spp:
    run(s, True, False)
    # if s >= 1:
    #     run(s, False, False)
