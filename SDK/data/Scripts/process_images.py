import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2
import numpy as np
import matplotlib.pyplot as plt

home_drive = "C:/Users/rpr/Desktop/meistdan"
base_dir = home_drive + "/optix/SDK/data/"

test_dir = os.path.join(base_dir, "test-jcgt-rev")

out_dir = os.path.join(base_dir, "jcgt-rev")
if not (os.path.exists(out_dir)):
    os.mkdir(out_dir)

lowres_dir = os.path.join(out_dir, "lowres")
if not (os.path.exists(lowres_dir)):
    os.mkdir(lowres_dir)

highres_dir = os.path.join(out_dir, "highres")
if not (os.path.exists(highres_dir)):
    os.mkdir(highres_dir)

p0 = "%.0f"
p2 = "%.2f"
s = "%.2e"
err_color_map = 'viridis'
density_color_map = 'inferno'


def read_image(filename):
    img = cv2.imread(filename, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    return img


def crop_image(img, ox, oy, w, h, bw, bwc, col):
    crop_img = img[oy: oy + h, ox: ox + w]
    if col[0] >= 0 and col[1] >= 0 and col[2] >= 0:
        crop_img[0: bwc, 0: w] = col
        crop_img[h - bwc: h, 0: w] = col
        crop_img[bwc: h - bwc, 0: bwc] = col
        crop_img[bwc: h - bwc, w - bwc: w] = col
        img[oy - bw: oy, ox - bw: ox + w + bw] = col
        img[oy + h: oy + h + bw, ox - bw: ox + w + bw] = col
        img[oy: oy + h, ox - bw: ox] = col
        img[oy: oy + h, ox + w: ox + w + bw] = col
    return img, crop_img


def mse(ref, img):
    return np.subtract(ref, img) ** 2


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


def falsecolor_bar(width, height, color_map, vertical):
    if vertical:
        scale = np.linspace(0, 1, height)
        scale = np.flip(scale)
        bar = np.tile(scale, (width, 1))
        bar = np.transpose(bar)
    else:
        scale = np.linspace(0, 1, width)
        bar = np.tile(scale, (height, 1))
    cmap = plt.get_cmap(color_map)
    bar = cmap(bar)
    bar = bar[:, :, [2, 1, 0]]
    return bar


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


def run(scene, bin_label, mdas_spp, mc_spp, morton_bit, extra_img_bit, scale_factor, alpha, gamma, rect0,
        rect1, col0, col1, bar_width, bar_height, vertical, ext, min_val, max_val, test_dir):
    # scene name extension
    print(scene)
    scene_ext = scene
    if ext:
        scene_ext = scene_ext + "-" + ext

    # reference image
    ref_test_name = scene
    ref_test_name += "-" + bin_label
    ref_test_name += "-reference"
    ref_filename = os.path.join(test_dir, ref_test_name + ".exr")
    ref_image = read_image(ref_filename)

    ref_test_name_denoised = ref_test_name + "-denoised"
    ref_filename_denoised = os.path.join(test_dir, ref_test_name_denoised + ".exr")
    ref_image_denoised = read_image(ref_filename_denoised)

    # mdas and mc
    scene = scene
    bin_label = bin_label
    mdas_test_name = scene + "-" + bin_label
    mdas_test_name += "-mdas"
    mdas_test_name += "-mb-" + str(morton_bit)
    mdas_test_name += "-eib-" + str(extra_img_bit)
    mdas_test_name += "-sf-" + str(scale_factor)
    mdas_test_name += "-et-" + str(alpha)
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

    # error
    mc_mse_image = mse(ref_image, mc_image)
    mc_mse = np.mean(mc_mse_image)
    mc_mse_image = falsecolor(mc_mse_image, err_color_map, min_val, mc_mse if max_val <= 0 else max_val)
    mdas_mse_image = mse(ref_image, mdas_image)
    mdas_mse = np.mean(mdas_mse_image)
    mdas_mse_image = falsecolor(mdas_mse_image, err_color_map, min_val, mc_mse if max_val <= 0 else max_val)
    ref_mse_image = mse(ref_image, ref_image)
    ref_mse = np.mean(ref_mse_image)
    ref_mse_image = falsecolor(ref_mse_image, err_color_map, min_val, mc_mse if max_val <= 0 else max_val)
    mc_relmse_image = rel_mse(ref_image, mc_image)
    mc_relmse = np.mean(mc_relmse_image.mean())
    mc_relmse_image = falsecolor(mc_relmse_image, err_color_map, min_val, mc_relmse if max_val <= 0 else max_val)
    mdas_relmse_image = rel_mse(ref_image, mdas_image)
    mdas_relmse = np.mean(mdas_relmse_image)
    mdas_relmse_image = falsecolor(mdas_relmse_image, err_color_map, min_val, mc_relmse if max_val <= 0 else max_val)
    ref_relmse_image = rel_mse(ref_image, ref_image)
    ref_relmse = np.mean(ref_relmse_image)
    ref_relmse_image = falsecolor(ref_relmse_image, err_color_map, min_val, mc_relmse if max_val <= 0 else max_val)

    mc_mse_image_denoised = mse(ref_image, mc_image_denoised)
    mc_mse_denoised = np.mean(mc_mse_image_denoised)
    mc_mse_image_denoised = falsecolor(mc_mse_image_denoised, err_color_map, min_val, mc_mse_denoised if max_val <= 0 else max_val)
    mdas_mse_image_denoised = mse(ref_image, mdas_image_denoised)
    mdas_mse_denoised = np.mean(mdas_mse_image_denoised)
    mdas_mse_image_denoised = falsecolor(mdas_mse_image_denoised, err_color_map, min_val, mc_mse_denoised if max_val <= 0 else max_val)
    ref_mse_image_denoised = mse(ref_image, ref_image_denoised)
    ref_mse_denoised = np.mean(ref_mse_image_denoised)
    ref_mse_image_denoised = falsecolor(ref_mse_image_denoised, err_color_map, min_val, mc_mse_denoised if max_val <= 0 else max_val)
    mc_relmse_image_denoised = rel_mse(ref_image, mc_image_denoised)
    mc_relmse_denoised = np.mean(mc_relmse_image_denoised.mean())
    mc_relmse_image_denoised = falsecolor(mc_relmse_image_denoised, err_color_map, min_val, mc_relmse_denoised if max_val <= 0 else max_val)
    mdas_relmse_image_denoised = rel_mse(ref_image, mdas_image_denoised)
    mdas_relmse_denoised = np.mean(mdas_relmse_image_denoised)
    mdas_relmse_image_denoised = falsecolor(mdas_relmse_image_denoised, err_color_map, min_val, mc_relmse_denoised if max_val <= 0 else max_val)
    ref_relmse_image_denoised = rel_mse(ref_image, ref_image_denoised)
    ref_relmse_denoised = np.mean(ref_relmse_image_denoised)
    ref_relmse_image_denoised = falsecolor(ref_relmse_image_denoised, err_color_map, min_val, mc_relmse_denoised if max_val <= 0 else max_val)

    # density colormap
    mdas_image_density_fc = falsecolor(mdas_image_density, density_color_map)

    # color bars
    err_bar = falsecolor_bar(bar_width, bar_height, err_color_map, vertical)
    density_bar = falsecolor_bar(bar_width, bar_height, density_color_map, vertical)

    # time and samples
    mdas_log_filename = os.path.join(test_dir, mdas_test_name + ".exr.log")
    mc_log_filename = os.path.join(test_dir, mc_test_name + ".exr.log")
    width = get_values("WIDTH", mc_log_filename)[0]
    height = get_values("HEIGHT", mc_log_filename)[0]
    mdas_total_samples = sum(get_values("TOTAL SAMPLES", mdas_log_filename))
    mdas_spp_avg = mdas_total_samples / (width * height)
    mdas_total_iterations = len(get_values("ADAPTIVE SAMPLING TIME", mdas_log_filename))

    mdas_initial_sampling_time = sum(get_values("INITIAL SAMPLING TIME", mdas_log_filename))
    mdas_construct_time = sum(get_values("CONSTRUCT TIME", mdas_log_filename))
    mdas_compute_errors_time = sum(get_values("COMPUTE ERRORS TIME", mdas_log_filename))
    mdas_adaptive_sampling_time = sum(get_values("ADAPTIVE SAMPLING TIME", mdas_log_filename))
    mdas_update_indices_time = sum(get_values("UPDATE INDICES TIME", mdas_log_filename))
    mdas_integrate_time = sum(get_values("INTEGRATE TIME", mdas_log_filename))
    mdas_trace_time = sum(get_values("TRACE TIME", mdas_log_filename))
    mdas_denoising_time = sum(get_values("DENOISING TIME", mdas_log_filename))

    mdas_total_time = 0
    mdas_total_time += mdas_initial_sampling_time
    mdas_total_time += mdas_construct_time
    mdas_total_time += mdas_compute_errors_time
    mdas_total_time += mdas_adaptive_sampling_time
    mdas_total_time += mdas_update_indices_time
    mdas_total_time += mdas_integrate_time
    mdas_total_time += mdas_trace_time

    mc_trace_time = sum(get_values("TRACE TIME", mc_log_filename))
    mc_denoising_time = sum(get_values("DENOISING TIME", mc_log_filename))
    mc_total_time = mc_trace_time

    # print time and error
    out_filename = scene_ext + ".tex"
    out_file = open(os.path.join(out_dir, out_filename), "w")
    out_file.write(mdas_test_name + "\n")
    out_file.write(str(p2 % mdas_spp_avg) + " samples per pixel\\\\\n")
    out_file.write("Render time " + str(p0 % mdas_total_time) + " ms\\\\\n")
    out_file.write("MSE " + str(s % mdas_mse) + " / ")
    out_file.write("RelMSE " + str(s % mdas_relmse) + "\n")
    out_file.write("Iterations " + str(mdas_total_iterations) + "\n")
    out_file.write("\n")
    out_file.write(mc_test_name + "\n")
    out_file.write(str(mc_spp) + " samples per pixel\\\\\n")
    out_file.write("Render time " + str(p0 % mc_total_time) + " ms\\\\\n")
    out_file.write("MSE " + str(s % mc_mse) + " / ")
    out_file.write("RelMSE " + str(s % mc_relmse) + "\n")
    out_file.write("\n")
    out_file.write(mdas_test_name + "-denoised\n")
    out_file.write(str(p2 % mdas_spp_avg) + " samples per pixel\\\\\n")
    out_file.write("Render time " + str(p0 % (mdas_total_time + mdas_denoising_time)) + " ms\\\\\n")
    out_file.write("MSE " + str(s % mdas_mse_denoised) + " / ")
    out_file.write("RelMSE " + str(s % mdas_relmse_denoised) + "\n")
    out_file.write("Iterations " + str(mdas_total_iterations) + "\n")
    out_file.write("\n")
    out_file.write(mc_test_name + "-denoised\n")
    out_file.write(str(mc_spp) + " samples per pixel\\\\\n")
    out_file.write("Render time " + str(p0 % (mc_total_time + mc_denoising_time)) + " ms\\\\\n")
    out_file.write("MSE " + str(s % mc_mse_denoised) + " / ")
    out_file.write("RelMSE " + str(s % mc_relmse_denoised) + "\n")
    out_file.write("\n")

    # hdr to ldr
    ref_image_ldr = 255 * np.clip(ref_image ** (1 / gamma), 0, 1)
    ref_image_denoised_ldr = 255 * np.clip(ref_image_denoised ** (1 / gamma), 0, 1)
    mdas_image_ldr = 255 * np.clip(mdas_image ** (1 / gamma), 0, 1)
    mc_image_ldr = 255 * np.clip(mc_image ** (1 / gamma), 0, 1)
    mdas_image_denoised_ldr = 255 * np.clip(mdas_image_denoised ** (1 / gamma), 0, 1)
    mc_image_denoised_ldr = 255 * np.clip(mc_image_denoised ** (1 / gamma), 0, 1)
    mdas_image_density_ldr = 255 * np.clip(mdas_image_density, 0, 1)
    mdas_image_density_fc_ldr = 255 * mdas_image_density_fc
    mdas_mse_image_ldr = 255 * mdas_mse_image
    mc_mse_image_ldr = 255 * mc_mse_image
    ref_mse_image_ldr = 255 * ref_mse_image
    mdas_relmse_image_ldr = 255 * mdas_relmse_image
    mc_relmse_image_ldr = 255 * mc_relmse_image
    ref_relmse_image_ldr = 255 * ref_relmse_image
    mdas_mse_image_denoised_ldr = 255 * mdas_mse_image_denoised
    mc_mse_image_denoised_ldr = 255 * mc_mse_image_denoised
    ref_mse_image_denoised_ldr = 255 * ref_mse_image_denoised
    mdas_relmse_image_denoised_ldr = 255 * mdas_relmse_image_denoised
    mc_relmse_image_denoised_ldr = 255 * mc_relmse_image_denoised
    ref_relmse_image_denoised_ldr = 255 * ref_relmse_image_denoised
    err_bar_ldr = 255 * err_bar
    density_bar_ldr = 255 * density_bar

    # crop
    bw = 7
    bwc = 2

    (ref_image_ldr, ref_image_ldr_closeup0) = crop_image(ref_image_ldr, rect0[0][0], rect0[0][1], rect0[1][0], rect0[1][1], bw, bwc, col0)
    (ref_image_ldr, ref_image_ldr_closeup1) = crop_image(ref_image_ldr, rect1[0][0], rect1[0][1], rect1[1][0], rect1[1][1], bw, bwc, col1)
    (mdas_image_ldr, mdas_image_ldr_closeup0) = crop_image(mdas_image_ldr, rect0[0][0], rect0[0][1], rect0[1][0], rect0[1][1], bw, bwc, col0)
    (mdas_image_ldr, mdas_image_ldr_closeup1) = crop_image(mdas_image_ldr, rect1[0][0], rect1[0][1], rect1[1][0], rect1[1][1], bw, bwc, col1)
    (mc_image_ldr, mc_image_ldr_closeup0) = crop_image(mc_image_ldr, rect0[0][0], rect0[0][1], rect0[1][0], rect0[1][1], bw, bwc, col0)
    (mc_image_ldr, mc_image_ldr_closeup1) = crop_image(mc_image_ldr, rect1[0][0], rect1[0][1], rect1[1][0], rect1[1][1], bw, bwc, col1)

    (ref_image_denoised_ldr, ref_image_denoised_ldr_closeup0) = crop_image(ref_image_denoised_ldr, rect0[0][0], rect0[0][1], rect0[1][0], rect0[1][1], bw, bwc, col0)
    (ref_image_denoised_ldr, ref_image_denoised_ldr_closeup1) = crop_image(ref_image_denoised_ldr, rect1[0][0], rect1[0][1], rect1[1][0], rect1[1][1], bw, bwc, col1)
    (mdas_image_denoised_ldr, mdas_image_denoised_ldr_closeup0) = crop_image(mdas_image_denoised_ldr, rect0[0][0], rect0[0][1], rect0[1][0], rect0[1][1], bw, bwc, col0)
    (mdas_image_denoised_ldr, mdas_image_denoised_ldr_closeup1) = crop_image(mdas_image_denoised_ldr, rect1[0][0], rect1[0][1], rect1[1][0], rect1[1][1], bw, bwc, col1)
    (mc_image_denoised_ldr, mc_image_denoised_ldr_closeup0) = crop_image(mc_image_denoised_ldr, rect0[0][0], rect0[0][1], rect0[1][0], rect0[1][1], bw, bwc, col0)
    (mc_image_denoised_ldr, mc_image_denoised_ldr_closeup1) = crop_image(mc_image_denoised_ldr, rect1[0][0], rect1[0][1], rect1[1][0], rect1[1][1], bw, bwc, col1)

    (ref_mse_image_ldr, ref_mse_image_ldr_closeup0) = crop_image(ref_mse_image_ldr, rect0[0][0], rect0[0][1], rect0[1][0], rect0[1][1], bw, bwc, col0)
    (ref_mse_image_ldr, ref_mse_image_ldr_closeup1) = crop_image(ref_mse_image_ldr, rect1[0][0], rect1[0][1], rect1[1][0], rect1[1][1], bw, bwc, col1)
    (mdas_mse_image_ldr, mdas_mse_image_ldr_closeup0) = crop_image(mdas_mse_image_ldr, rect0[0][0], rect0[0][1], rect0[1][0], rect0[1][1], bw, bwc, col0)
    (mdas_mse_image_ldr, mdas_mse_image_ldr_closeup1) = crop_image(mdas_mse_image_ldr, rect1[0][0], rect1[0][1], rect1[1][0], rect1[1][1], bw, bwc, col1)
    (mc_mse_image_ldr, mc_mse_image_ldr_closeup0) = crop_image(mc_mse_image_ldr, rect0[0][0], rect0[0][1], rect0[1][0], rect0[1][1], bw, bwc, col0)
    (mc_mse_image_ldr, mc_mse_image_ldr_closeup1) = crop_image(mc_mse_image_ldr, rect1[0][0], rect1[0][1], rect1[1][0], rect1[1][1], bw, bwc, col1)

    (ref_relmse_image_ldr, ref_relmse_image_ldr_closeup0) = crop_image(ref_relmse_image_ldr, rect0[0][0], rect0[0][1], rect0[1][0], rect0[1][1], bw, bwc, col0)
    (ref_relmse_image_ldr, ref_relmse_image_ldr_closeup1) = crop_image(ref_relmse_image_ldr, rect1[0][0], rect1[0][1], rect1[1][0], rect1[1][1], bw, bwc, col1)
    (mdas_relmse_image_ldr, mdas_relmse_image_ldr_closeup0) = crop_image(mdas_relmse_image_ldr, rect0[0][0], rect0[0][1], rect0[1][0], rect0[1][1], bw, bwc, col0)
    (mdas_relmse_image_ldr, mdas_relmse_image_ldr_closeup1) = crop_image(mdas_relmse_image_ldr, rect1[0][0], rect1[0][1], rect1[1][0], rect1[1][1], bw, bwc, col1)
    (mc_relmse_image_ldr, mc_relmse_image_ldr_closeup0) = crop_image(mc_relmse_image_ldr, rect0[0][0], rect0[0][1], rect0[1][0], rect0[1][1], bw, bwc, col0)
    (mc_relmse_image_ldr, mc_relmse_image_ldr_closeup1) = crop_image(mc_relmse_image_ldr, rect1[0][0], rect1[0][1], rect1[1][0], rect1[1][1], bw, bwc, col1)

    (ref_mse_image_denoised_ldr, ref_mse_image_denoised_ldr_closeup0) = crop_image(ref_mse_image_denoised_ldr, rect0[0][0], rect0[0][1], rect0[1][0], rect0[1][1], bw, bwc, col0)
    (ref_mse_image_denoised_ldr, ref_mse_image_denoised_ldr_closeup1) = crop_image(ref_mse_image_denoised_ldr, rect1[0][0], rect1[0][1], rect1[1][0], rect1[1][1], bw, bwc, col1)
    (mdas_mse_image_denoised_ldr, mdas_mse_image_denoised_ldr_closeup0) = crop_image(mdas_mse_image_denoised_ldr, rect0[0][0], rect0[0][1], rect0[1][0], rect0[1][1], bw, bwc, col0)
    (mdas_mse_image_denoised_ldr, mdas_mse_image_denoised_ldr_closeup1) = crop_image(mdas_mse_image_denoised_ldr, rect1[0][0], rect1[0][1], rect1[1][0], rect1[1][1], bw, bwc, col1)
    (mc_mse_image_denoised_ldr, mc_mse_image_denoised_ldr_closeup0) = crop_image(mc_mse_image_denoised_ldr, rect0[0][0], rect0[0][1], rect0[1][0], rect0[1][1], bw, bwc, col0)
    (mc_mse_image_denoised_ldr, mc_mse_image_denoised_ldr_closeup1) = crop_image(mc_mse_image_denoised_ldr, rect1[0][0], rect1[0][1], rect1[1][0], rect1[1][1], bw, bwc, col1)

    (ref_relmse_image_denoised_ldr, ref_relmse_image_denoised_ldr_closeup0) = crop_image(ref_relmse_image_denoised_ldr, rect0[0][0], rect0[0][1], rect0[1][0], rect0[1][1], bw, bwc, col0)
    (ref_relmse_image_denoised_ldr, ref_relmse_image_denoised_ldr_closeup1) = crop_image(ref_relmse_image_denoised_ldr, rect1[0][0], rect1[0][1], rect1[1][0], rect1[1][1], bw, bwc, col1)
    (mdas_relmse_image_denoised_ldr, mdas_relmse_image_denoised_ldr_closeup0) = crop_image(mdas_relmse_image_denoised_ldr, rect0[0][0], rect0[0][1], rect0[1][0], rect0[1][1], bw, bwc, col0)
    (mdas_relmse_image_denoised_ldr, mdas_relmse_image_denoised_ldr_closeup1) = crop_image(mdas_relmse_image_denoised_ldr, rect1[0][0], rect1[0][1], rect1[1][0], rect1[1][1], bw, bwc, col1)
    (mc_relmse_image_denoised_ldr, mc_relmse_image_denoised_ldr_closeup0) = crop_image(mc_relmse_image_denoised_ldr, rect0[0][0], rect0[0][1], rect0[1][0], rect0[1][1], bw, bwc, col0)
    (mc_relmse_image_denoised_ldr, mc_relmse_image_denoised_ldr_closeup1) = crop_image(mc_relmse_image_denoised_ldr, rect1[0][0], rect1[0][1], rect1[1][0], rect1[1][1], bw, bwc, col1)

    (mdas_image_density_ldr, mdas_image_density_ldr_closeup0) = crop_image(mdas_image_density_ldr, rect0[0][0], rect0[0][1], rect0[1][0], rect0[1][1], bw, bwc, col0)
    (mdas_image_density_ldr, mdas_image_density_ldr_closeup1) = crop_image(mdas_image_density_ldr, rect1[0][0], rect1[0][1], rect1[1][0], rect1[1][1], bw, bwc, col1)
    (mdas_image_density_fc_ldr, mdas_image_density_fc_ldr_closeup0) = crop_image(mdas_image_density_fc_ldr, rect0[0][0], rect0[0][1], rect0[1][0], rect0[1][1], bw, bwc, col0)
    (mdas_image_density_fc_ldr, mdas_image_density_fc_ldr_closeup1) = crop_image(mdas_image_density_fc_ldr, rect1[0][0], rect1[0][1], rect1[1][0], rect1[1][1], bw, bwc, col1)

    # write highres ldr
    cv2.imwrite(os.path.join(highres_dir, scene_ext + "-ref.png"), ref_image_ldr)
    cv2.imwrite(os.path.join(highres_dir, scene_ext + "-mdas.png"), mdas_image_ldr)
    cv2.imwrite(os.path.join(highres_dir, scene_ext + "-mc.png"), mc_image_ldr)
    cv2.imwrite(os.path.join(highres_dir, scene_ext + "-ref-denoised.png"), ref_image_denoised_ldr)
    cv2.imwrite(os.path.join(highres_dir, scene_ext + "-mdas-denoised.png"), mdas_image_denoised_ldr)
    cv2.imwrite(os.path.join(highres_dir, scene_ext + "-mc-denoised.png"), mc_image_denoised_ldr)
    cv2.imwrite(os.path.join(highres_dir, scene_ext + "-ref-mse.png"), ref_mse_image_ldr)
    cv2.imwrite(os.path.join(highres_dir, scene_ext + "-mdas-mse.png"), mdas_mse_image_ldr)
    cv2.imwrite(os.path.join(highres_dir, scene_ext + "-mc-mse.png"), mc_mse_image_ldr)
    cv2.imwrite(os.path.join(highres_dir, scene_ext + "-ref-relmse.png"), ref_relmse_image_ldr)
    cv2.imwrite(os.path.join(highres_dir, scene_ext + "-mdas-relmse.png"), mdas_relmse_image_ldr)
    cv2.imwrite(os.path.join(highres_dir, scene_ext + "-mc-relmse.png"), mc_relmse_image_ldr)
    cv2.imwrite(os.path.join(highres_dir, scene_ext + "-ref-mse-denoised.png"), ref_mse_image_denoised_ldr)
    cv2.imwrite(os.path.join(highres_dir, scene_ext + "-mdas-mse-denoised.png"), mdas_mse_image_denoised_ldr)
    cv2.imwrite(os.path.join(highres_dir, scene_ext + "-mc-mse-denoised.png"), mc_mse_image_denoised_ldr)
    cv2.imwrite(os.path.join(highres_dir, scene_ext + "-ref-relmse-denoised.png"), ref_relmse_image_denoised_ldr)
    cv2.imwrite(os.path.join(highres_dir, scene_ext + "-mdas-relmse-denoised.png"), mdas_relmse_image_denoised_ldr)
    cv2.imwrite(os.path.join(highres_dir, scene_ext + "-mc-relmse-denoised.png"), mc_relmse_image_denoised_ldr)
    cv2.imwrite(os.path.join(highres_dir, scene_ext + "-mdas-density.png"), mdas_image_density_ldr)
    cv2.imwrite(os.path.join(highres_dir, scene_ext + "-mdas-density-fc.png"), mdas_image_density_fc_ldr)
    cv2.imwrite(os.path.join(highres_dir, scene_ext + "-closeup0-ref.png"), ref_image_ldr_closeup0)
    cv2.imwrite(os.path.join(highres_dir, scene_ext + "-closeup1-ref.png"), ref_image_ldr_closeup1)
    cv2.imwrite(os.path.join(highres_dir, scene_ext + "-closeup0-mdas.png"), mdas_image_ldr_closeup0)
    cv2.imwrite(os.path.join(highres_dir, scene_ext + "-closeup1-mdas.png"), mdas_image_ldr_closeup1)
    cv2.imwrite(os.path.join(highres_dir, scene_ext + "-closeup0-mc.png"), mc_image_ldr_closeup0)
    cv2.imwrite(os.path.join(highres_dir, scene_ext + "-closeup1-mc.png"), mc_image_ldr_closeup1)
    cv2.imwrite(os.path.join(highres_dir, scene_ext + "-closeup0-ref-denoised.png"), ref_image_denoised_ldr_closeup0)
    cv2.imwrite(os.path.join(highres_dir, scene_ext + "-closeup1-ref-denoised.png"), ref_image_denoised_ldr_closeup1)
    cv2.imwrite(os.path.join(highres_dir, scene_ext + "-closeup0-mdas-denoised.png"), mdas_image_denoised_ldr_closeup0)
    cv2.imwrite(os.path.join(highres_dir, scene_ext + "-closeup1-mdas-denoised.png"), mdas_image_denoised_ldr_closeup1)
    cv2.imwrite(os.path.join(highres_dir, scene_ext + "-closeup0-mc-denoised.png"), mc_image_denoised_ldr_closeup0)
    cv2.imwrite(os.path.join(highres_dir, scene_ext + "-closeup1-mc-denoised.png"), mc_image_denoised_ldr_closeup1)
    cv2.imwrite(os.path.join(highres_dir, scene_ext + "-closeup0-ref-mse.png"), ref_mse_image_ldr_closeup0)
    cv2.imwrite(os.path.join(highres_dir, scene_ext + "-closeup1-ref-mse.png"), ref_mse_image_ldr_closeup1)
    cv2.imwrite(os.path.join(highres_dir, scene_ext + "-closeup0-mdas-mse.png"), mdas_mse_image_ldr_closeup0)
    cv2.imwrite(os.path.join(highres_dir, scene_ext + "-closeup1-mdas-mse.png"), mdas_mse_image_ldr_closeup1)
    cv2.imwrite(os.path.join(highres_dir, scene_ext + "-closeup0-mc-mse.png"), mc_mse_image_ldr_closeup0)
    cv2.imwrite(os.path.join(highres_dir, scene_ext + "-closeup1-mc-mse.png"), mc_mse_image_ldr_closeup1)
    cv2.imwrite(os.path.join(highres_dir, scene_ext + "-closeup0-ref-relmse.png"), ref_relmse_image_ldr_closeup0)
    cv2.imwrite(os.path.join(highres_dir, scene_ext + "-closeup1-ref-relmse.png"), ref_relmse_image_ldr_closeup1)
    cv2.imwrite(os.path.join(highres_dir, scene_ext + "-closeup0-mdas-relmse.png"), mdas_relmse_image_ldr_closeup0)
    cv2.imwrite(os.path.join(highres_dir, scene_ext + "-closeup1-mdas-relmse.png"), mdas_relmse_image_ldr_closeup1)
    cv2.imwrite(os.path.join(highres_dir, scene_ext + "-closeup0-mc-relmse.png"), mc_relmse_image_ldr_closeup0)
    cv2.imwrite(os.path.join(highres_dir, scene_ext + "-closeup1-mc-relmse.png"), mc_relmse_image_ldr_closeup1)
    cv2.imwrite(os.path.join(highres_dir, scene_ext + "-closeup0-ref-mse-denoised.png"), ref_mse_image_denoised_ldr_closeup0)
    cv2.imwrite(os.path.join(highres_dir, scene_ext + "-closeup1-ref-mse-denoised.png"), ref_mse_image_denoised_ldr_closeup1)
    cv2.imwrite(os.path.join(highres_dir, scene_ext + "-closeup0-mdas-mse-denoised.png"), mdas_mse_image_denoised_ldr_closeup0)
    cv2.imwrite(os.path.join(highres_dir, scene_ext + "-closeup1-mdas-mse-denoised.png"), mdas_mse_image_denoised_ldr_closeup1)
    cv2.imwrite(os.path.join(highres_dir, scene_ext + "-closeup0-mc-mse-denoised.png"), mc_mse_image_denoised_ldr_closeup0)
    cv2.imwrite(os.path.join(highres_dir, scene_ext + "-closeup1-mc-mse-denoised.png"), mc_mse_image_denoised_ldr_closeup1)
    cv2.imwrite(os.path.join(highres_dir, scene_ext + "-closeup0-ref-relmse-denoised.png"), ref_relmse_image_denoised_ldr_closeup0)
    cv2.imwrite(os.path.join(highres_dir, scene_ext + "-closeup1-ref-relmse-denoised.png"), ref_relmse_image_denoised_ldr_closeup1)
    cv2.imwrite(os.path.join(highres_dir, scene_ext + "-closeup0-mdas-relmse-denoised.png"), mdas_relmse_image_denoised_ldr_closeup0)
    cv2.imwrite(os.path.join(highres_dir, scene_ext + "-closeup1-mdas-relmse-denoised.png"), mdas_relmse_image_denoised_ldr_closeup1)
    cv2.imwrite(os.path.join(highres_dir, scene_ext + "-closeup0-mc-relmse-denoised.png"), mc_relmse_image_denoised_ldr_closeup0)
    cv2.imwrite(os.path.join(highres_dir, scene_ext + "-closeup1-mc-relmse-denoised.png"), mc_relmse_image_denoised_ldr_closeup1)
    cv2.imwrite(os.path.join(highres_dir, scene_ext + "-closeup0-mdas-density.png"), mdas_image_density_ldr_closeup0)
    cv2.imwrite(os.path.join(highres_dir, scene_ext + "-closeup1-mdas-density.png"), mdas_image_density_ldr_closeup1)
    cv2.imwrite(os.path.join(highres_dir, scene_ext + "-closeup0-mdas-density-fc.png"), mdas_image_density_fc_ldr_closeup0)
    cv2.imwrite(os.path.join(highres_dir, scene_ext + "-closeup1-mdas-density-fc.png"), mdas_image_density_fc_ldr_closeup1)
    cv2.imwrite(os.path.join(highres_dir, scene_ext + "-err-bar.png"), err_bar_ldr)
    cv2.imwrite(os.path.join(highres_dir, scene_ext + "-density-bar.png"), density_bar_ldr)

    # resize
    max_edge = max(width, height)
    scale = 512 / max_edge
    dim_lowres = (int(width * scale), int(height * scale))
    ref_image_ldr_lowres = cv2.resize(ref_image_ldr, dim_lowres)
    mdas_image_ldr_lowres = cv2.resize(mdas_image_ldr, dim_lowres)
    mc_image_ldr_lowres = cv2.resize(mc_image_ldr, dim_lowres)
    ref_image_denoised_ldr_lowres = cv2.resize(ref_image_denoised_ldr, dim_lowres)
    mdas_image_denoised_ldr_lowres = cv2.resize(mdas_image_denoised_ldr, dim_lowres)
    mc_image_denoised_ldr_lowres = cv2.resize(mc_image_denoised_ldr, dim_lowres)
    mdas_image_density_ldr_lowres = cv2.resize(mdas_image_density_ldr, dim_lowres)
    mdas_image_density_fc_ldr_lowres = cv2.resize(mdas_image_density_fc_ldr, dim_lowres)
    ref_mse_image_ldr_lowres = cv2.resize(ref_mse_image_ldr, dim_lowres)
    mdas_mse_image_ldr_lowres = cv2.resize(mdas_mse_image_ldr, dim_lowres)
    mc_mse_image_ldr_lowres = cv2.resize(mc_mse_image_ldr, dim_lowres)
    ref_relmse_image_ldr_lowres = cv2.resize(ref_relmse_image_ldr, dim_lowres)
    mdas_relmse_image_ldr_lowres = cv2.resize(mdas_relmse_image_ldr, dim_lowres)
    mc_relmse_image_ldr_lowres = cv2.resize(mc_relmse_image_ldr, dim_lowres)
    ref_mse_image_denoised_ldr_lowres = cv2.resize(ref_mse_image_denoised_ldr, dim_lowres)
    mdas_mse_image_denoised_ldr_lowres = cv2.resize(mdas_mse_image_denoised_ldr, dim_lowres)
    mc_mse_image_denoised_ldr_lowres = cv2.resize(mc_mse_image_denoised_ldr, dim_lowres)
    ref_relmse_image_denoised_ldr_lowres = cv2.resize(ref_relmse_image_denoised_ldr, dim_lowres)
    mdas_relmse_image_denoised_ldr_lowres = cv2.resize(mdas_relmse_image_denoised_ldr, dim_lowres)
    mc_relmse_image_denoised_ldr_lowres = cv2.resize(mc_relmse_image_denoised_ldr, dim_lowres)

    # write lowres ldr
    cv2.imwrite(os.path.join(lowres_dir, scene_ext + "-ref.png"), ref_image_ldr_lowres)
    cv2.imwrite(os.path.join(lowres_dir, scene_ext + "-mdas.png"), mdas_image_ldr_lowres)
    cv2.imwrite(os.path.join(lowres_dir, scene_ext + "-mc.png"), mc_image_ldr_lowres)
    cv2.imwrite(os.path.join(lowres_dir, scene_ext + "-ref-denoised.png"), ref_image_denoised_ldr_lowres)
    cv2.imwrite(os.path.join(lowres_dir, scene_ext + "-mdas-denoised.png"), mdas_image_denoised_ldr_lowres)
    cv2.imwrite(os.path.join(lowres_dir, scene_ext + "-mc-denoised.png"), mc_image_denoised_ldr_lowres)
    cv2.imwrite(os.path.join(lowres_dir, scene_ext + "-ref-mse.png"), ref_mse_image_ldr_lowres)
    cv2.imwrite(os.path.join(lowres_dir, scene_ext + "-mdas-mse.png"), mdas_mse_image_ldr_lowres)
    cv2.imwrite(os.path.join(lowres_dir, scene_ext + "-mc-mse.png"), mc_mse_image_ldr_lowres)
    cv2.imwrite(os.path.join(lowres_dir, scene_ext + "-ref-relmse.png"), ref_relmse_image_ldr_lowres)
    cv2.imwrite(os.path.join(lowres_dir, scene_ext + "-mdas-relmse.png"), mdas_relmse_image_ldr_lowres)
    cv2.imwrite(os.path.join(lowres_dir, scene_ext + "-mc-relmse.png"), mc_relmse_image_ldr_lowres)
    cv2.imwrite(os.path.join(lowres_dir, scene_ext + "-ref-mse-denoised.png"), ref_mse_image_denoised_ldr_lowres)
    cv2.imwrite(os.path.join(lowres_dir, scene_ext + "-mdas-mse-denoised.png"), mdas_mse_image_denoised_ldr_lowres)
    cv2.imwrite(os.path.join(lowres_dir, scene_ext + "-mc-mse-denoised.png"), mc_mse_image_denoised_ldr_lowres)
    cv2.imwrite(os.path.join(lowres_dir, scene_ext + "-ref-relmse-denoised.png"), ref_relmse_image_denoised_ldr_lowres)
    cv2.imwrite(os.path.join(lowres_dir, scene_ext + "-mdas-relmse-denoised.png"), mdas_relmse_image_denoised_ldr_lowres)
    cv2.imwrite(os.path.join(lowres_dir, scene_ext + "-mc-relmse-denoised.png"), mc_relmse_image_denoised_ldr_lowres)
    cv2.imwrite(os.path.join(lowres_dir, scene_ext + "-mdas-density.png"), mdas_image_density_ldr_lowres)
    cv2.imwrite(os.path.join(lowres_dir, scene_ext + "-mdas-density-fc.png"), mdas_image_density_fc_ldr_lowres)
    cv2.imwrite(os.path.join(lowres_dir, scene_ext + "-closeup0-ref.png"), ref_image_ldr_closeup0)
    cv2.imwrite(os.path.join(lowres_dir, scene_ext + "-closeup1-ref.png"), ref_image_ldr_closeup1)
    cv2.imwrite(os.path.join(lowres_dir, scene_ext + "-closeup0-mdas.png"), mdas_image_ldr_closeup0)
    cv2.imwrite(os.path.join(lowres_dir, scene_ext + "-closeup1-mdas.png"), mdas_image_ldr_closeup1)
    cv2.imwrite(os.path.join(lowres_dir, scene_ext + "-closeup0-mc.png"), mc_image_ldr_closeup0)
    cv2.imwrite(os.path.join(lowres_dir, scene_ext + "-closeup1-mc.png"), mc_image_ldr_closeup1)
    cv2.imwrite(os.path.join(lowres_dir, scene_ext + "-closeup0-ref-denoised.png"), ref_image_denoised_ldr_closeup0)
    cv2.imwrite(os.path.join(lowres_dir, scene_ext + "-closeup1-ref-denoised.png"), ref_image_denoised_ldr_closeup1)
    cv2.imwrite(os.path.join(lowres_dir, scene_ext + "-closeup0-mdas-denoised.png"), mdas_image_denoised_ldr_closeup0)
    cv2.imwrite(os.path.join(lowres_dir, scene_ext + "-closeup1-mdas-denoised.png"), mdas_image_denoised_ldr_closeup1)
    cv2.imwrite(os.path.join(lowres_dir, scene_ext + "-closeup0-mc-denoised.png"), mc_image_denoised_ldr_closeup0)
    cv2.imwrite(os.path.join(lowres_dir, scene_ext + "-closeup1-mc-denoised.png"), mc_image_denoised_ldr_closeup1)
    cv2.imwrite(os.path.join(lowres_dir, scene_ext + "-closeup0-ref-mse.png"), ref_mse_image_ldr_closeup0)
    cv2.imwrite(os.path.join(lowres_dir, scene_ext + "-closeup1-ref-mse.png"), ref_mse_image_ldr_closeup1)
    cv2.imwrite(os.path.join(lowres_dir, scene_ext + "-closeup0-mdas-mse.png"), mdas_mse_image_ldr_closeup0)
    cv2.imwrite(os.path.join(lowres_dir, scene_ext + "-closeup1-mdas-mse.png"), mdas_mse_image_ldr_closeup1)
    cv2.imwrite(os.path.join(lowres_dir, scene_ext + "-closeup0-mc-mse.png"), mc_mse_image_ldr_closeup0)
    cv2.imwrite(os.path.join(lowres_dir, scene_ext + "-closeup1-mc-mse.png"), mc_mse_image_ldr_closeup1)
    cv2.imwrite(os.path.join(lowres_dir, scene_ext + "-closeup0-ref-relmse.png"), ref_relmse_image_ldr_closeup0)
    cv2.imwrite(os.path.join(lowres_dir, scene_ext + "-closeup1-ref-relmse.png"), ref_relmse_image_ldr_closeup1)
    cv2.imwrite(os.path.join(lowres_dir, scene_ext + "-closeup0-mdas-relmse.png"), mdas_relmse_image_ldr_closeup0)
    cv2.imwrite(os.path.join(lowres_dir, scene_ext + "-closeup1-mdas-relmse.png"), mdas_relmse_image_ldr_closeup1)
    cv2.imwrite(os.path.join(lowres_dir, scene_ext + "-closeup0-mc-relmse.png"), mc_relmse_image_ldr_closeup0)
    cv2.imwrite(os.path.join(lowres_dir, scene_ext + "-closeup1-mc-relmse.png"), mc_relmse_image_ldr_closeup1)
    cv2.imwrite(os.path.join(lowres_dir, scene_ext + "-closeup0-ref-mse-denoised.png"), ref_mse_image_denoised_ldr_closeup0)
    cv2.imwrite(os.path.join(lowres_dir, scene_ext + "-closeup1-ref-mse-denoised.png"), ref_mse_image_denoised_ldr_closeup1)
    cv2.imwrite(os.path.join(lowres_dir, scene_ext + "-closeup0-mdas-mse-denoised.png"), mdas_mse_image_denoised_ldr_closeup0)
    cv2.imwrite(os.path.join(lowres_dir, scene_ext + "-closeup1-mdas-mse-denoised.png"), mdas_mse_image_denoised_ldr_closeup1)
    cv2.imwrite(os.path.join(lowres_dir, scene_ext + "-closeup0-mc-mse-denoised.png"), mc_mse_image_denoised_ldr_closeup0)
    cv2.imwrite(os.path.join(lowres_dir, scene_ext + "-closeup1-mc-mse-denoised.png"), mc_mse_image_denoised_ldr_closeup1)
    cv2.imwrite(os.path.join(lowres_dir, scene_ext + "-closeup0-ref-relmse-denoised.png"), ref_relmse_image_denoised_ldr_closeup0)
    cv2.imwrite(os.path.join(lowres_dir, scene_ext + "-closeup1-ref-relmse-denoised.png"), ref_relmse_image_denoised_ldr_closeup1)
    cv2.imwrite(os.path.join(lowres_dir, scene_ext + "-closeup0-mdas-relmse-denoised.png"), mdas_relmse_image_denoised_ldr_closeup0)
    cv2.imwrite(os.path.join(lowres_dir, scene_ext + "-closeup1-mdas-relmse-denoised.png"), mdas_relmse_image_denoised_ldr_closeup1)
    cv2.imwrite(os.path.join(lowres_dir, scene_ext + "-closeup0-mc-relmse-denoised.png"), mc_relmse_image_denoised_ldr_closeup0)
    cv2.imwrite(os.path.join(lowres_dir, scene_ext + "-closeup1-mc-relmse-denoised.png"), mc_relmse_image_denoised_ldr_closeup1)
    cv2.imwrite(os.path.join(lowres_dir, scene_ext + "-closeup0-mdas-density.png"), mdas_image_density_ldr_closeup0)
    cv2.imwrite(os.path.join(lowres_dir, scene_ext + "-closeup1-mdas-density.png"), mdas_image_density_ldr_closeup1)
    cv2.imwrite(os.path.join(lowres_dir, scene_ext + "-closeup0-mdas-density-fc.png"), mdas_image_density_fc_ldr_closeup0)
    cv2.imwrite(os.path.join(lowres_dir, scene_ext + "-closeup1-mdas-density-fc.png"), mdas_image_density_fc_ldr_closeup1)
    cv2.imwrite(os.path.join(lowres_dir, scene_ext + "-err-bar.png"), err_bar_ldr)
    cv2.imwrite(os.path.join(lowres_dir, scene_ext + "-density-bar.png"), density_bar_ldr)


# pool-mb-mdas-mb-1-eib-8-sf-0.0625-et-0.25-spp-4
run("pool", "mb", 4, 6, 1, 8, 0.0625, 0.25, 2.2, [[364, 340], [50, 50]], [[234, 757], [50, 50]], [35, 51, 239], [70, 187, 95], 48, 512, True, "", 0, 5.0e-3, test_dir)

# chess-dof-mdas-mb-1-eib-8-sf-1-et-0.25-spp-8
run("chess", "dof", 8, 11, 1, 8, 1, 0.25, 2.2, [[40, 604], [90, 90]], [[590, 355], [75, 75]], [35, 51, 239], [70, 187, 95], 48, 528, True, "", 0, 1.0e-2, test_dir)

# Bistro-dof-mdas-mb-0-eib-10-sf-0.0625-et-0.03125-spp-8
run("Bistro", "dof", 8, 9, 0, 10, 0.0625, 0.03125, 2.2, [[990, 488], [100, 56]], [[934, 773], [120, 68]], [35, 51, 239], [70, 187, 95], 48, 648, True, "", 0, 1.0e-2, test_dir)

# cornell-box-pt-mdas-mb-1-eib-7-sf-1-et-0.0625-spp-8
run("cornell-box", "pt", 8, 6, 1, 7, 1, 0.0625, 2.2, [[540, 775], [100, 100]], [[280, 880], [60, 60]], [35, 51, 239], [70, 187, 95], 48, 512, True, "", 0, 1.0e-1, test_dir)

# breakfast-pt-mdas-mb-1-eib-8-sf-0.5-et-0.015625-spp-8
run("breakfast", "pt", 8, 8, 1, 8, 0.5, 0.015625, 2.2, [[585, 655], [100, 75]], [[245, 335], [80, 60]], [35, 51, 239], [70, 187, 95], 48, 864, True, "", 0, 1.0e-1, test_dir)

# dragon-dl-mdas-mb-0-eib-10-sf-0.25-et-0.015625-spp-8
run("dragon", "dl", 8, 13, 0, 10, 0.25, 0.015625, 2.2, [[140, 510], [100, 75]], [[585, 270], [40, 30]], [35, 51, 239], [70, 187, 95], 48, 864, True, "", 0, 3.0e-2, test_dir)

# cobblestone-dl-mdas-mb-0-eib-10-sf-0.0625-et-0.015625-spp-4
run("cobblestone", "dl", 4, 6, 0, 10, 0.0625, 0.015625, 2.2, [[1553, 304], [100, 56]], [[1275, 908], [120, 68]], [35, 51, 239], [70, 187, 95], 48, 648, True, "", 0, 5.0e-2, test_dir)

# pool-mb-mdas-mb-1-eib-8-sf-0.0625-et-0.25-spp-4
run("pool", "mb", 4, 6, 1, 8, 0.0625, 0.25, 2.2, [[686, 334], [50, 50]], [[400, 450], [50, 50]], [35, 51, 239], [70, 187, 95], 48, 864, True, "denoising", 0, 1.0e-2, test_dir)

# cornell-box-pt-mdas-mb-1-eib-7-sf-1-et-0.0625-spp-8
# cornell-box-pt-mdas-mb-2-eib-4-sf-0.0625-et-0.0625-spp-8
run("cornell-box", "pt", 8, 8, 1, 7, 1, 0.0625, 2.2, [[150, 160], [100, 100]], [[250, 430], [60, 60]], [35, 51, 239], [70, 187, 95], 48, 432, True, "scale-0", 0, 1.0e-2, test_dir)
run("cornell-box", "pt", 8, 8, 2, 4, 0.0625, 0.0625, 2.2, [[150, 160], [100, 100]], [[250, 430], [60, 60]], [35, 51, 239], [70, 187, 95], 48, 432, True, "scale-1", 0, 1.0e-2, test_dir)
