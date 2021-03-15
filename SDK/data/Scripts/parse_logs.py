import os
import cv2
import numpy as np

home_drive = "C:/Users/meist/projects"
base_dir = home_drive + "/optix/SDK/data/"

out_dir = os.path.join(base_dir, "test")
if not (os.path.exists(out_dir)):
    os.mkdir(out_dir)
os.chdir(out_dir)

# table file
table_filename = os.path.join(out_dir, "table.csv")
if os.path.exists(table_filename):
    os.remove(table_filename)

testing_passes = 5

# pool-mb-mdas-mb-1-eib-9-sf-0.5-et-0.01-spp-4-pass-0.exr.log
#pool-mb-spp-4-pass-0.exr.log

scenes = ["pool", "chess", "Bistro", "picapica", "san-miguel", "gallery", "crytek-sponza", "hairball", "cornell-box", "picapica", "dragon"]
bin_labels = ["mb", "dof", "ao", "pt"]
bin_indices = [0, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3]
scene_indices = [0, 1, 2, 4, 7]

# spps = [0.25, 0.5, 1, 2, 4]
spps = [4]
extra_img_bits = [8, 8, 9]
morton_bits = [0, 1, 0]
scale_factors = [0.25, 0.5, 1.0]
error_thresholds = [0.01, 0.025, 0.05]

assert(len(morton_bits) == len(extra_img_bits))
bits_num = len(morton_bits)

p = "%.3f"


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


def run(scene_index, spp, morton_bit, extra_img_bit,  scale_factor, error_threshold, mdas):

    # test name
    scene = scenes[scene_index]
    bin_index = bin_indices[scene_index]
    bin_label = bin_labels[bin_index]
    test_name = scene + "-" + bin_label
    if mdas:
        test_name += "-mdas"
        test_name += "-mb-" + str(morton_bit)
        test_name += "-eib-" + str(extra_img_bit)
        test_name += "-sf-" + str(scale_factor)
        test_name += "-et-" + str(error_threshold)
    test_name += "-spp-" + str(spp)

    table_file = open(os.path.join(out_dir, table_filename), "a")
    table_file.write(test_name)

    total_samples = 0
    total_iterations = 0
    initial_sampling_time = 0
    construct_time = 0
    compute_errors_time = 0
    adaptive_sampling_time = 0
    update_indices_time = 0
    integrate_time = 0
    denoising_time = 0
    trace_time = 0
    error = 0
    error_denoised = 0

    for testing_pass in range(testing_passes):

        # test name
        test_name_pass = test_name + "-pass-" + str(testing_pass)

        # log and image filename
        image_filename = os.path.join(out_dir, test_name_pass + ".exr")
        image_denoised_filename = os.path.join(out_dir, test_name_pass + "-denoised.exr")
        log_filename = os.path.join(out_dir, test_name_pass + ".exr.log")

        # parse
        print(test_name_pass)

        width = get_values("WIDTH", log_filename)[0]
        height = get_values("HEIGHT", log_filename)[0]
        if mdas:
            total_samples += sum(get_values("TOTAL SAMPLES", log_filename))
        else:
            total_samples += width * height * spp
        total_iterations += max(len(get_values("ADAPTIVE SAMPLING TIME", log_filename)) - 2, 0)
        initial_sampling_time += sum(get_values("INITIAL SAMPLING TIME", log_filename))
        construct_time += sum(get_values("CONSTRUCT TIME", log_filename))
        compute_errors_time += sum(get_values("COMPUTE ERRORS TIME", log_filename))
        adaptive_sampling_time += sum(get_values("ADAPTIVE SAMPLING TIME", log_filename))
        update_indices_time += sum(get_values("UPDATE INDICES TIME", log_filename))
        integrate_time += sum(get_values("INTEGRATE TIME", log_filename))
        denoising_time += sum(get_values("DENOISING TIME", log_filename))
        trace_time += sum(get_values("TRACE TIME", log_filename))

        image = read_image(image_filename)
        image_denoised = read_image(image_denoised_filename)
        error += mse(ref_image, image)
        error_denoised += mse(ref_image, image_denoised)

    error /= testing_passes
    error_denoised /= testing_passes

    total_samples /= testing_passes
    total_iterations /= testing_passes
    initial_sampling_time /= testing_passes
    construct_time /= testing_passes
    compute_errors_time /= testing_passes
    adaptive_sampling_time /= testing_passes
    update_indices_time /= testing_passes
    integrate_time /= testing_passes
    denoising_time /= testing_passes
    trace_time /= testing_passes

    total_mdas_time = 0
    total_mdas_time += initial_sampling_time
    total_mdas_time += construct_time
    total_mdas_time += compute_errors_time
    total_mdas_time += adaptive_sampling_time
    total_mdas_time += update_indices_time
    total_mdas_time += integrate_time

    total_time = total_mdas_time + trace_time
    total_time_denoising = total_time + denoising_time

    table_file.write(", " + str(p % int(total_samples)))
    table_file.write(", " + str(p % int(total_iterations)))
    table_file.write(", " + str(p % initial_sampling_time))
    table_file.write(", " + str(p % construct_time))
    table_file.write(", " + str(p % compute_errors_time))
    table_file.write(", " + str(p % adaptive_sampling_time))
    table_file.write(", " + str(p % update_indices_time))
    table_file.write(", " + str(p % integrate_time))
    table_file.write(", " + str(p % denoising_time))
    table_file.write(", " + str(p % total_mdas_time))
    table_file.write(", " + str(p % trace_time))
    table_file.write(", " + str(p % total_time))
    table_file.write(", " + str(error))
    table_file.write(", " + str(p % total_time_denoising))
    table_file.write(", " + str(error_denoised))
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
table_file.write(", adaptive sampling time")
table_file.write(", update indices time")
table_file.write(", integrate time")
table_file.write(", denoising time")
table_file.write(", total mdas time")
table_file.write(", trace time")
table_file.write(", total time")
table_file.write(", mse")
table_file.write(", total time (denoised)")
table_file.write(", mse (denoised)")
table_file.write("\n")
table_file.close()

# scene_index, spp, morton_bit, extra_img_bit,  scale_factor, error_threshold, mdas
for scene_index in scene_indices:

    ref_test_name = scenes[scene_index]
    ref_test_name += "-" + bin_labels[bin_indices[scene_index]]
    ref_test_name += "-reference"
    ref_filename = ref_test_name
    ref_filename = os.path.join(out_dir, ref_test_name + ".exr")
    ref_image = read_image(ref_filename)

    for spp in spps:
        if spp >= 1:
            run(scene_index, spp, 0, 0, 0, 0, False)

    for spp in spps:
        for bit_index in range(bits_num):
            for scale_factor in scale_factors:
                for error_threshold in error_thresholds:
                    morton_bit = morton_bits[bit_index]
                    extra_img_bit = extra_img_bits[bit_index]
                    run(scene_index, spp, morton_bit, extra_img_bit, scale_factor, error_threshold, True)