import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

two_bounces = True

home_drive = "C:/Users/meist/projects"
base_dir = home_drive + "/optix/SDK/data/"

test_dir = os.path.join(base_dir, "test")

out_dir = os.path.join(base_dir, "jcgt")
if not (os.path.exists(out_dir)):
    os.mkdir(out_dir)

supplementary_dir = os.path.join(out_dir, "supplementary")
if not (os.path.exists(supplementary_dir)):
    os.mkdir(supplementary_dir)

p0 = "%.0f"
p2 = "%.2f"
s = "%.2e"
err_color_map = 'viridis'


def read_image(filename):
    img = cv2.imread(filename, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    return img


def rel_mse(ref, img):
    return (np.subtract(ref, img) ** 2) / ((ref ** 2) + 1.0e-2)


def falsecolor(img, color_map, min_val=0, max_val=1):
    cmap = plt.get_cmap(color_map)
    if img.ndim > 2:
        img = np.mean(img, axis=2)
    img = np.clip((img - min_val) / (max_val - min_val + 1.0e-2), 0, 1)
    img = cmap(img)
    img = img[:, :, [2, 1, 0]]
    return img


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


def run(scene, bin_label, mdas_spp, mc_spp, morton_bit, extra_img_bit, scale_factor, error_threshold, gamma, ext, min_val, max_val, test_dir):

    # scene name extension
    scene_ext = scene
    if ext:
        scene_ext = scene_ext + "-" + ext

    # reference image
    ref_test_name = scene
    ref_test_name += "-" + bin_label
    ref_test_name += "-reference"
    ref_filename = os.path.join(test_dir, ref_test_name + ".exr")
    ref_image = read_image(ref_filename)

    # mdas and mc
    scene = scene
    bin_label = bin_label
    mdas_test_name = scene + "-" + bin_label
    mdas_test_name += "-mdas"
    mdas_test_name += "-mb-" + str(morton_bit)
    mdas_test_name += "-eib-" + str(extra_img_bit)
    mdas_test_name += "-sf-" + str(scale_factor)
    mdas_test_name += "-et-" + str(error_threshold)
    mdas_test_name += "-spp-" + str(mdas_spp)
    mdas_test_name += "-pass-" + str(0)
    mc_test_name = scene + "-" + bin_label
    mc_test_name += "-spp-" + str(mc_spp)
    mc_test_name += "-pass-" + str(0)

    # images
    mdas_image_filename = os.path.join(test_dir, mdas_test_name + ".exr")
    mdas_image = read_image(mdas_image_filename)
    mc_image_filename = os.path.join(test_dir, mc_test_name + ".exr")
    mc_image = read_image(mc_image_filename)
    mdas_image_denoised_filename = os.path.join(test_dir, mdas_test_name + "-denoised.exr")
    mdas_image_denoised = read_image(mdas_image_denoised_filename)
    mc_image_denoised_filename = os.path.join(test_dir, mc_test_name + "-denoised.exr")
    mc_image_denoised = read_image(mc_image_denoised_filename)
    mdas_image_density_filename = os.path.join(test_dir, mdas_test_name + "-density.exr")
    mdas_image_density = read_image(mdas_image_density_filename)

    mdas_image[np.isnan(mdas_image)] = 0
    mc_image[np.isnan(mc_image)] = 0
    mdas_image_denoised[np.isnan(mdas_image_denoised)] = 0
    mc_image_denoised[np.isnan(mc_image_denoised)] = 0
    mdas_image_density[np.isnan(mdas_image_density)] = 0

    # write hdr
    cv2.imwrite(os.path.join(supplementary_dir, scene_ext + "-ref.exr"), ref_image)
    cv2.imwrite(os.path.join(supplementary_dir, scene_ext + "-mdas.exr"), mdas_image)
    cv2.imwrite(os.path.join(supplementary_dir, scene_ext + "-mc.exr"), mc_image)
    cv2.imwrite(os.path.join(supplementary_dir, scene_ext + "-mdas-denoised.exr"), mdas_image_denoised)
    cv2.imwrite(os.path.join(supplementary_dir, scene_ext + "-mc-denoised.exr"), mc_image_denoised)
    cv2.imwrite(os.path.join(supplementary_dir, scene_ext + "-mdas-density.exr"), mdas_image_density)

    # error
    mc_relmse_image = rel_mse(ref_image, mc_image)
    mc_relmse = np.mean(mc_relmse_image.mean())
    mc_relmse_image = falsecolor(mc_relmse_image, err_color_map, min_val, mc_relmse if max_val <= 0 else max_val)
    mdas_relmse_image = rel_mse(ref_image, mdas_image)
    mdas_relmse_image = falsecolor(mdas_relmse_image, err_color_map, min_val, mc_relmse if max_val <= 0 else max_val)

    mc_relmse_image_denoised = rel_mse(ref_image, mc_image_denoised)
    mc_relmse_denoised = np.mean(mc_relmse_image_denoised.mean())
    mc_relmse_image_denoised = falsecolor(mc_relmse_image_denoised, err_color_map, min_val, mc_relmse if max_val <= 0 else max_val)
    mdas_relmse_image_denoised = rel_mse(ref_image, mdas_image_denoised)
    mdas_relmse_image_denoised = falsecolor(mdas_relmse_image_denoised, err_color_map, min_val, mc_relmse if max_val <= 0 else max_val)

    # hdr to ldr
    ref_image_ldr = 255 * np.clip(ref_image ** (1 / gamma), 0, 1)
    mdas_image_ldr = 255 * np.clip(mdas_image ** (1 / gamma), 0, 1)
    mc_image_ldr = 255 * np.clip(mc_image ** (1 / gamma), 0, 1)
    mdas_image_denoised_ldr = 255 * np.clip(mdas_image_denoised ** (1 / gamma), 0, 1)
    mc_image_denoised_ldr = 255 * np.clip(mc_image_denoised ** (1 / gamma), 0, 1)
    mdas_image_density_ldr = 255 * np.clip(mdas_image_density, 0, 1)
    mdas_relmse_image_ldr = 255 * mdas_relmse_image
    mc_relmse_image_ldr = 255 * mc_relmse_image
    mdas_relmse_image_denoised_ldr = 255 * mdas_relmse_image_denoised
    mc_relmse_image_denoised_ldr = 255 * mc_relmse_image_denoised

    # write ldr
    cv2.imwrite(os.path.join(supplementary_dir, scene_ext + "-ref.png"), ref_image_ldr)
    cv2.imwrite(os.path.join(supplementary_dir, scene_ext + "-mdas.png"), mdas_image_ldr)
    cv2.imwrite(os.path.join(supplementary_dir, scene_ext + "-mc.png"), mc_image_ldr)
    cv2.imwrite(os.path.join(supplementary_dir, scene_ext + "-mdas-denoised.png"), mdas_image_denoised_ldr)
    cv2.imwrite(os.path.join(supplementary_dir, scene_ext + "-mc-denoised.png"), mc_image_denoised_ldr)
    cv2.imwrite(os.path.join(supplementary_dir, scene_ext + "-mdas-relmse.png"), mdas_relmse_image_ldr)
    cv2.imwrite(os.path.join(supplementary_dir, scene_ext + "-mc-relmse.png"), mc_relmse_image_ldr)
    cv2.imwrite(os.path.join(supplementary_dir, scene_ext + "-mdas-relmse-denoised.png"), mdas_relmse_image_denoised_ldr)
    cv2.imwrite(os.path.join(supplementary_dir, scene_ext + "-mc-relmse-denoised.png"), mc_relmse_image_denoised_ldr)
    cv2.imwrite(os.path.join(supplementary_dir, scene_ext + "-mdas-density.png"), mdas_image_density_ldr)


# scene, bin_label, mdas_spp, mc_spp, morton_bit, extra_img_bit, scale_factor, error_threshold, gamma, ext, min_val, max_val, test_dir):

# pool-mb-mdas-mb-1-eib-8-sf-0.0625-et-0.25-spp-4
run("pool", "mb", 4, 6, 1, 8, 0.0625, 0.25, 2.2, "", 0, 5.0e-3, test_dir)

# chess-dof-mdas-mb-1-eib-8-sf-1-et-0.25-spp-8
run("chess", "dof", 8, 10, 1, 8, 1, 0.25, 2.2, "", 0, 1.0e-2, test_dir)

# Bistro-dof-mdas-mb-0-eib-10-sf-0.0625-et-0.03125-spp-8
run("Bistro", "dof", 8, 8, 0, 10, 0.0625, 0.03125, 2.2, "", 0, 1.0e-2, test_dir)

# cornell-box-pt-mdas-mb-1-eib-7-sf-1-et-0.0625-spp-8
run("cornell-box", "pt", 8, 7, 1, 7, 1, 0.0625, 2.2, "", 0, 1.0e-1, test_dir)

# breakfast-pt-mdas-mb-1-eib-8-sf-0.5-et-0.015625-spp-8
run("breakfast", "pt", 8, 7, 1, 8, 0.5, 0.015625, 2.2, "", 0, 1.0e-1, test_dir)

# dragon-dl-mdas-mb-0-eib-10-sf-0.25-et-0.015625-spp-8
run("dragon", "dl", 8, 13, 0, 10, 0.25, 0.015625, 2.2, "", 0, 3.0e-2, test_dir)

# pool-mb-mdas-mb-1-eib-8-sf-0.0625-et-0.25-spp-4
run("pool", "mb", 8, 15, 1, 8, 0.0625, 0.25, 2.2, "denoising", 0, 1.0e-2, test_dir)

# cornell-box-pt-mdas-mb-1-eib-7-sf-1-et-0.0625-spp-8
# cornell-box-pt-mdas-mb-2-eib-4-sf-0.0625-et-0.0625-spp-8
run("cornell-box", "pt", 8, 8, 1, 7, 1, 0.0625, 2.2, "scale-0", 0, 1.0e-2, test_dir)
run("cornell-box", "pt", 8, 8, 2, 4, 0.0625, 0.0625, 2.2, "scale-1", 0, 1.0e-2, test_dir)