import os

home_drive = "C:/Users/rpr/Desktop/meistdan"
optix_bin_dir = home_drive + "/optix/SDK/build/bin/Release/"
base_dir = home_drive + "/optix/SDK/data/"

out_dir = os.path.join(base_dir, "test-jcgt-rev")
if not (os.path.exists(out_dir)):
    os.mkdir(out_dir)
os.chdir(optix_bin_dir)

ref_enabled = True
mc_enabled = True
mdas_enabled = True
testing_passes = 1

scenes = ["pool", "chess", "Bistro", "picapica", "san-miguel", "gallery", "crytek-sponza", "hairball", "cornell-box", "picapica", "dragon", "breakfast", "cornell-box", "cobblestone", "hairball"]
bins = ["optixMotionBlur.exe", "optixDepthOfField.exe", "optixAmbientOcclusion.exe", "optixPathTracer.exe", "optixDirectLighting.exe"]
bin_labels = ["mb", "dof", "ao", "pt", "dl"]
bin_indices = [0, 1, 1, 2, 2, 2, 2, 2, 3, 3, 4, 3, 4, 4, 4]
ref_spp = 65536

# scene_indices = [0, 1, 2, 10, 11]
# scene_indices = [8]
scene_indices = [13]
mc_spps = list(range(1, 33))
spps = [4, 8]
extra_img_bits = [7, 8, 10]
morton_bits = [1, 1, 0]
scale_factors = [1/16, 1/4, 1/2, 1]
alphas = [1/64, 1/32, 1/16, 1/4]
# extra_img_bits = [4]
# morton_bits = [2]
# scale_factors = [1/16]
# alphas = [1/16]

assert(len(morton_bits) == len(extra_img_bits))
bits_num = len(morton_bits)


def run(scene, bin, bin_label, spp, morton_bit, extra_img_bit,  scale_factor, alpha, testing_pass, mdas, ref):

    # test name
    scene = scene
    test_name = scene + "-" + bin_label
    if ref:
        test_name += "-reference"
    else:
        if mdas:
            test_name += "-mdas"
            test_name += "-mb-" + str(morton_bit)
            test_name += "-eib-" + str(extra_img_bit)
            test_name += "-sf-" + str(scale_factor)
            test_name += "-et-" + str(alpha)
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
    test_file.write("Sampler {\n")
    test_file.write("mdas " + ("true" if mdas else "false") + "\n")
    test_file.write("samples " + str(spp) + "\n")
    test_file.write("}\n")
    test_file.write("\n")

    # mdas
    if mdas:
        test_file.write("Mdas {\n")
        test_file.write("scaleFactor " + str(scale_factor) + "\n")
        test_file.write("alpha " + str(alpha) + "\n")
        test_file.write("bitsPerDim " + str(morton_bit) + "\n")
        test_file.write("extraImgBits " + str(extra_img_bit) + "\n")
        test_file.write("}\n")
        test_file.write("\n")

    # append scene definition
    test_file.write(scene_data)
    test_file.close()

    # binary
    bin_full = os.path.join(optix_bin_dir, bin)

    # execute
    print(test_name)
    os.system(bin_full + " " + test_filename_full)


# scene_index, spp, morton_bit, extra_img_bit,  scale_factor, error_threshold, testing_pass, mdas, ref
for scene_index in scene_indices:

    scene = scenes[scene_index]
    bin_index = bin_indices[scene_index]
    bin_label = bin_labels[bin_index]
    bin = bins[bin_index]

    if ref_enabled:
        spp = ref_spp
        run(scene, bin, bin_label, spp, 0, 0, 0, 0, 0, False, True)

    for testing_pass in range(testing_passes):
        if mc_enabled:
            for spp in mc_spps:
                run(scene, bin, bin_label, spp, 0, 0, 1, 0, testing_pass, False, False)

        if mdas_enabled:
            for spp in spps:
                for scale_factor in scale_factors:
                    for alpha in alphas:
                        for bit_index in range(bits_num):
                            morton_bit = morton_bits[bit_index]
                            extra_img_bit = extra_img_bits[bit_index]
                            run(scene, bin, bin_label, spp, morton_bit, extra_img_bit, scale_factor, alpha, testing_pass,
                                True, False)
