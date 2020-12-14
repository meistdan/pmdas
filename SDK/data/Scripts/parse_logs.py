import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

scene_name = "Bistro"

home_drive = "C:/Users/meist/projects"
optix_exe = home_drive + "/optix/SDK/build/bin/Release/optixDepthOfField.exe"
base_dir = home_drive + "/optix/SDK/data/" + scene_name + "/"

# output dir
out_dir = os.path.join(base_dir, "mdas")
if not (os.path.exists(out_dir)):
    os.mkdir(out_dir)
os.chdir(base_dir)

# stats file
stats_filename = os.path.join(out_dir, "stats.txt")
if os.path.exists(stats_filename):
    os.remove(stats_filename)

width = 1920
height = 1080

ref_spp = 1024
spp = [0.25, 0.5, 1, 2, 4]


def read_image(filename):
    img = cv2.imread(filename)
    return img


def mse(img0: np.array, img1 : np.array):
    return (np.subtract(img0, img1) ** 2).mean()


def get_values(key, filename):
    f = open(filename)
    res = []
    key = key + "\n"
    while True:
        line = f.readline()
        if not line: break
        if key in line:
            res.append(float(f.readline()))
    f.close()
    return res


def run(spp, mdas):
    # test name
    test_name = scene_name
    if mdas:
        test_name += "-mdas"
    test_name += "-spp-" + str(spp)

    # log and image filename
    image_filename = os.path.join(out_dir, test_name + ".exr")
    log_filename = os.path.join(out_dir, test_name + ".exr.log")

    # parse
    print(test_name)
    total_time = 0
    stats_file = open(os.path.join(out_dir, stats_filename), "a")
    stats_file.write(test_name + "\n")
    if mdas:
        total_samples = sum(get_values("TOTAL SAMPLES", log_filename))
        stats_file.write("TOTAL SAMPLES\t" + str(total_samples) + "\n")

        total_iterations = len(get_values("SAMPLES", log_filename)) - 2
        stats_file.write("TOTAL ITERATIONS\t" + str(total_iterations) + "\n")

        initial_sampling_time = sum(get_values("INITIAL SAMPLING TIME", log_filename))
        stats_file.write("INITIAL SAMPLING TIME\t" + str(initial_sampling_time) + "\n")
        total_time += initial_sampling_time

        construct_time = sum(get_values("CONSTRUCT TIME", log_filename))
        stats_file.write("CONSTRUCT TIME\t" + str(construct_time) + "\n")
        total_time += construct_time

        compute_errors_time = sum(get_values("COMPUTE ERRORS TIME", log_filename))
        stats_file.write("COMPUTE ERRORS TIME\t" + str(compute_errors_time) + "\n")
        total_time += compute_errors_time

        adaptive_sampling_time = sum(get_values("ADAPTIVE SAMPLING TIME", log_filename))
        stats_file.write("ADAPTIVE SAMPLING TIME\t" + str(adaptive_sampling_time) + "\n")
        total_time += adaptive_sampling_time

        split_time = sum(get_values("SPLIT TIME", log_filename))
        stats_file.write("SPLIT TIME\t" + str(split_time) + "\n")
        total_time += split_time

        prepare_leaf_indices_time = sum(get_values("PREPARE LEAF INDICES TIME", log_filename))
        stats_file.write("PREPARE LEAF INDICES TIME\t" + str(prepare_leaf_indices_time) + "\n")
        total_time += prepare_leaf_indices_time

        update_indices_time = sum(get_values("UPDATE INDICES TIME", log_filename))
        stats_file.write("UPDATE INDICES TIME\t" + str(update_indices_time) + "\n")
        total_time += update_indices_time

        integrate_time = sum(get_values("INTEGRATE TIME", log_filename))
        stats_file.write("INTEGRATE TIME\t" + str(integrate_time) + "\n")
        total_time += integrate_time

        stats_file.write("TOTAL MDAS TIME\t" + str(total_time) + "\n")
    else:
        total_samples = width * height * spp
        stats_file.write("TOTAL SAMPLES\t" + str(total_samples) + "\n")
    trace_time = sum(get_values("TRACE TIME", log_filename))
    stats_file.write("TRACE TIME\t" + str(trace_time) + "\n")
    total_time += trace_time
    stats_file.write("TOTAL TIME\t" + str(total_time) + "\n")
    image = read_image(image_filename)
    error = mse(ref_image, image)
    stats_file.write("MSE\t" + str(error) + "\n")
    stats_file.write("\n")
    stats_file.close()


# ref test name
ref_test_name = scene_name
ref_test_name += "-reference"
ref_filename = ref_test_name
ref_filename = os.path.join(out_dir, ref_test_name + ".exr")
ref_image = read_image(ref_filename)

for s in spp:
    run(s, True)
    if s >= 1:
        run(s, False)
