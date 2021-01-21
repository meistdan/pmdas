import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

scene_name = "Bistro"

home_drive = "C:/Users/meist/projects"
optix_exe = home_drive + "/optix/SDK/build/bin/Release/optixDepthOfField.exe"
base_dir = home_drive + "/optix/SDK/data/" + scene_name + "/"

# output dir
out_dir = os.path.join(base_dir, "mdas-error-per-node")
if not (os.path.exists(out_dir)):
    os.mkdir(out_dir)
os.chdir(base_dir)

# table file
table_filename = os.path.join(out_dir, "table.csv")
if os.path.exists(table_filename):
    os.remove(table_filename)

width = 1920
height = 1080

ref_spp = 1024
# spp = [0.25, 0.5, 1, 2, 4]
spp = [4]

p = "%.2f"


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
    table_file = open(os.path.join(out_dir, table_filename), "a")
    table_file.write(test_name)

    if mdas:
        total_samples = sum(get_values("TOTAL SAMPLES", log_filename))
    else:
        total_samples = width * height * spp
    table_file.write(", " + str(p % total_samples))

    total_iterations = max(len(get_values("SAMPLES", log_filename)) - 2, 0)
    table_file.write(", " + str(p % total_iterations))

    initial_sampling_time = sum(get_values("INITIAL SAMPLING TIME", log_filename))
    table_file.write(", " + str(p % initial_sampling_time))
    total_time += initial_sampling_time

    construct_time = sum(get_values("CONSTRUCT TIME", log_filename))
    table_file.write(", " + str(p % construct_time))
    total_time += construct_time

    compute_errors_time = sum(get_values("COMPUTE ERRORS TIME", log_filename))
    table_file.write(", " + str(p % compute_errors_time))
    total_time += compute_errors_time

    propagate_errors_time = sum(get_values("PROPAGATE ERRORS TIME", log_filename))
    table_file.write(", " + str(p % propagate_errors_time))
    total_time += propagate_errors_time

    adaptive_sampling_time = sum(get_values("ADAPTIVE SAMPLING TIME", log_filename))
    table_file.write(", " + str(p % adaptive_sampling_time))
    total_time += adaptive_sampling_time

    split_time = sum(get_values("SPLIT TIME", log_filename))
    table_file.write(", " + str(p % split_time))
    total_time += split_time

    update_indices_time = sum(get_values("UPDATE INDICES TIME", log_filename))
    table_file.write(", " + str(p % update_indices_time))
    total_time += update_indices_time

    integrate_time = sum(get_values("INTEGRATE TIME", log_filename))
    table_file.write(", " + str(p % integrate_time))
    total_time += integrate_time

    table_file.write(", " + str(p % total_time))

    trace_time = sum(get_values("TRACE TIME", log_filename))
    table_file.write(", " + str(p % trace_time))

    total_time += trace_time
    table_file.write(", " + str(p % total_time))

    image = read_image(image_filename)
    error = mse(ref_image, image)
    table_file.write(", " + str(error))

    table_file.write("\n")
    table_file.close()


# table
table_file = open(table_filename, "a")
table_file.write("method / stat")
table_file.write(", total samples")
table_file.write(", iterations")
table_file.write(", initial sampling time")
table_file.write(", construct time")
table_file.write(", compute errors time")
table_file.write(", propagate errors time")
table_file.write(", adaptive sampling time")
table_file.write(", split time")
table_file.write(", update indices time")
table_file.write(", integrate time")
table_file.write(", total mdas time")
table_file.write(", trace time")
table_file.write(", total time")
table_file.write(", mse")
table_file.write("\n")
table_file.close()

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
