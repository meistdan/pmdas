import os

home_drive = "C:/Users/meist/projects"
optix_bin_dir = home_drive + "/optix/SDK/build/bin/Release/"
base_dir = home_drive + "/optix/SDK/data/"

out_dir = os.path.join(base_dir, "test")
if not (os.path.exists(out_dir)):
    os.mkdir(out_dir)
os.chdir(optix_bin_dir)

ref_enabled = True
testing_passes = 5

scenes = ["pool", "chess", "Bistro", "picapica", "san-miguel", "gallery", "crytek-sponza", "hairball", "cornell-box", "picapica", "dragon"]
bins = ["optixMotionBlur.exe", "optixDepthOfField.exe", "optixAmbientOcclusion.exe", "optixPathTracer.exe"]
bin_labels = ["mb", "dof", "ao", "pt"]
bin_indices = [0, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3]
ref_spps = [1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 8192, 8192, 8192]
scene_indices = [0, 1, 2, 4, 7, 8, 10]

# spps = [0.25, 0.5, 1, 2, 4]
spps = [4]
extra_img_bits = [8, 8, 9, 10]
morton_bits = [0, 1, 0, 0]
scale_factors = [0.125, 0.25]
error_thresholds = [0.005, 0.01]

assert(len(morton_bits) == len(extra_img_bits))
bits_num = len(morton_bits)


def run(scene_index, spp, morton_bit, extra_img_bit,  scale_factor, error_threshold, testing_pass, mdas, ref):

    # test name
    scene = scenes[scene_index]
    bin_index = bin_indices[scene_index]
    bin_label = bin_labels[bin_index]
    test_name = scene + "-" + bin_label
    if ref:
        test_name += "-reference"
    else:
        if mdas:
            test_name += "-mdas"
            test_name += "-mb-" + str(morton_bit)
            test_name += "-eib-" + str(extra_img_bit)
            test_name += "-sf-" + str(scale_factor)
            test_name += "-et-" + str(error_threshold)
        test_name += "-spp-" + str(spp)
        test_name += "-pass-" + str(testing_pass)

    # scene definition
    scene_dir = os.path.join(base_dir, scene)
    scene_desc = open(os.path.join(scene_dir, scene + ".env"), "r")
    scene_data = scene_desc.read()
    scene_desc.close()

    # test file
    test_filename = test_name + ".env"
    test_filename_full = os.path.join(out_dir, test_filename)
    test_file = open(test_filename_full, "w")

    # film
    image_filename = os.path.join(out_dir, test_name + ".exr")
    test_file.write("Film {\n")
    test_file.write("filename " + image_filename + "\n")
    test_file.write("}\n")
    test_file.write("\n")

    # sample
    ref_spp = ref_spps[scene_index]
    test_file.write("Sampler {\n")
    test_file.write("mdas " + ("true" if mdas else "false") + "\n")
    test_file.write("samples " + (str(ref_spp) if ref else str(spp)) + "\n")
    test_file.write("}\n")
    test_file.write("\n")

    # mdas
    if mdas:
        test_file.write("Mdas {\n")
        test_file.write("scaleFactor " + str(scale_factor) + "\n")
        test_file.write("errorThreshold " + str(error_threshold) + "\n")
        test_file.write("bitsPerDim " + str(morton_bit) + "\n")
        test_file.write("extraImgBits " + str(extra_img_bit) + "\n")
        test_file.write("}\n")
        test_file.write("\n")

    # append scene definition
    test_file.write(scene_data)
    test_file.close()

    # binary
    bin = bins[bin_index]
    bin_full = os.path.join(optix_bin_dir, bin)

    # execute
    print(test_name)
    os.system(bin_full + " " + test_filename_full)


# scene_index, spp, morton_bit, extra_img_bit,  scale_factor, error_threshold, testing_pass, mdas, ref
for scene_index in scene_indices:

    if ref_enabled:
        run(scene_index, 0, 0, 0, 0, 0, 0, False, True)

    for testing_pass in range(testing_passes):
        for spp in spps:

            if spp >= 1:
                run(scene_index, spp, 0, 0, 1, 0, testing_pass, False, False)

            for scale_factor in scale_factors:
                for error_threshold in error_thresholds:
                    for bit_index in range(bits_num):
                        morton_bit = morton_bits[bit_index]
                        extra_img_bit = extra_img_bits[bit_index]
                        run(scene_index, spp, morton_bit, extra_img_bit, scale_factor, error_threshold, testing_pass,
                            True, False)
