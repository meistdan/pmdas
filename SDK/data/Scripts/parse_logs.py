import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2
import numpy as np

home_drive = "C:/Users/rpr/Desktop/meistdan"
base_dir = home_drive + "/optix/SDK/data/"

out_dir = os.path.join(base_dir, "test-jcgt-rev")
os.chdir(out_dir)

testing_passes = 1

scenes = ["pool", "chess", "Bistro", "picapica", "san-miguel", "gallery", "crytek-sponza", "hairball", "cornell-box", "picapica", "dragon", "breakfast", "cornell-box", "cobblestone", "hairball"]
bin_labels = ["mb", "dof", "ao", "pt", "dl"]
bin_indices = [0, 1, 1, 2, 2, 2, 2, 2, 3, 3, 4, 3, 4, 4, 4]

# scene_indices = [1, 2, 10, 11]
scene_indices = [13]
mc_spps = list(range(1, 33))
spps = [4, 8]
# spps = [4]
# extra_img_bits = [4, 6, 8, 10]
# morton_bits = [3, 2, 1, 0]
# extra_img_bits = [4, 7, 10]
# morton_bits = [2, 1, 0]
extra_img_bits = [7, 8, 10]
morton_bits = [1, 1, 0]
scale_factors = [1/16, 1/4, 1/2, 1]
alphas = [1/64, 1/32, 1/16, 1/4]

assert(len(morton_bits) == len(extra_img_bits))
bits_num = len(morton_bits)

p = "%.3f"


def read_image(filename):
    img = cv2.imread(filename, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    return img


def mse(ref, img):
    img[np.isnan(img)] = 0
    return (np.subtract(ref, img) ** 2).mean()


def rel_mse(ref, img):
    img[np.isnan(img)] = 0
    return ((np.subtract(ref, img) ** 2) / ((ref ** 2) + 1.0e-2)).mean()


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


def run(scene, bin_label, spp, morton_bit, extra_img_bit, scale_factor, alpha, mdas):

    # test name
    test_name = scene + "-" + bin_label
    if mdas:
        test_name += "-mdas"
        test_name += "-mb-" + str(morton_bit)
        test_name += "-eib-" + str(extra_img_bit)
        test_name += "-sf-" + str(scale_factor)
        test_name += "-et-" + str(alpha)
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
    mse_er = 0
    mse_er_denoised = 0
    relmse_er = 0
    relmse_er_denoised = 0

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
        total_iterations += len(get_values("ADAPTIVE SAMPLING TIME", log_filename))
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
        mse_er += mse(ref_image, image)
        mse_er_denoised += mse(ref_image, image_denoised)
        relmse_er += rel_mse(ref_image, image)
        relmse_er_denoised += rel_mse(ref_image, image_denoised)

    mse_er /= testing_passes
    mse_er_denoised /= testing_passes
    relmse_er /= testing_passes
    relmse_er_denoised /= testing_passes

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

    table_file.write(", " + str(p % total_time))
    table_file.write(", " + str(mse_er))
    table_file.write(", " + str(mse_er * total_time))
    table_file.write(", " + str(relmse_er))
    table_file.write(", " + str(relmse_er * total_time))
    table_file.write(", " + str(p % total_time_denoising))
    table_file.write(", " + str(mse_er_denoised))
    table_file.write(", " + str(mse_er_denoised * total_time_denoising))
    table_file.write(", " + str(relmse_er_denoised))
    table_file.write(", " + str(relmse_er_denoised * total_time_denoising))
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
    table_file.write("\n")
    table_file.close()


# scene_index, spp, morton_bit, extra_img_bit,  scale_factor, alpha, mdas
for scene_index in scene_indices:

    scene = scenes[scene_index]
    bin_index = bin_indices[scene_index]
    bin_label = bin_labels[bin_index]

    # table file
    table_filename = os.path.join(out_dir, "table-" + scene + "-" + bin_label + ".csv")
    if os.path.exists(table_filename):
        os.remove(table_filename)

    # table
    table_file = open(table_filename, "a")
    table_file.write("method / stat")
    table_file.write(", total time")
    table_file.write(", mse")
    table_file.write(", time * mse")
    table_file.write(", rmse")
    table_file.write(", time * rmse")
    table_file.write(", total time (denoised)")
    table_file.write(", mse (denoised)")
    table_file.write(", time * mse (denoised)")
    table_file.write(", rmse (denoised)")
    table_file.write(", time * rmse (denoised)")
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
    table_file.write("\n")
    table_file.close()

    ref_test_name = scene
    ref_test_name += "-" + bin_labels[bin_indices[scene_index]]
    ref_test_name += "-reference"
    ref_filename = ref_test_name
    ref_filename = os.path.join(out_dir, ref_test_name + ".exr")
    ref_image = read_image(ref_filename)

    for spp in mc_spps:
        run(scene, bin_label, spp, 0, 0, 0, 0, False)

    for spp in spps:
        for alpha in alphas:
            for bit_index in range(bits_num):
                for scale_factor in scale_factors:
                    morton_bit = morton_bits[bit_index]
                    extra_img_bit = extra_img_bits[bit_index]
                    run(scene, bin_label, spp, morton_bit, extra_img_bit, scale_factor, alpha, True)
