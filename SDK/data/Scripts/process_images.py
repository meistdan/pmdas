import os
import cv2
import numpy as np

home_drive = "C:/Users/meist/projects"
base_dir = home_drive + "/optix/SDK/data/"

test_dir = os.path.join(base_dir, "test")
os.chdir(test_dir)

out_dir = os.path.join(test_dir, "images")
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


def read_image(filename):
    img = cv2.imread(filename, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    return img


def crop_image(img, ox, oy, w, h, bw, bwc, col):
    crop_img = img[oy: oy + h, ox: ox + w]
    crop_img[0: bwc, 0: w] = col
    crop_img[h - bwc: h, 0: w] = col
    crop_img[bwc: h - bwc, 0: bwc] = col
    crop_img[bwc: h - bwc, w - bwc: w] = col
    img[oy - bw: oy, ox - bw: ox + w + bw] = col
    img[oy + h: oy + h + bw, ox - bw: ox + w + bw] = col
    img[oy: oy + h, ox - bw: ox] = col
    img[oy: oy + h, ox + w: ox + w + bw] = col
    return img, crop_img


def mse(img0, img1):
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


def run(scene, bin_label, mdas_spp, mc_spp, morton_bit, extra_img_bit, scale_factor, error_threshold, gamma, rect0, rect1):

    # reference image
    ref_test_name = scene
    ref_test_name += "-" + bin_label
    ref_test_name += "-reference"
    ref_filename = ref_test_name
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

    # error
    mdas_error = mse(ref_image, mdas_image)
    mc_error = mse(ref_image, mc_image)

    # time and samples
    mdas_log_filename = os.path.join(test_dir, mdas_test_name + ".exr.log")
    mc_log_filename = os.path.join(test_dir, mc_test_name + ".exr.log")
    width = get_values("WIDTH", mc_log_filename)[0]
    height = get_values("HEIGHT", mc_log_filename)[0]
    mdas_total_samples = sum(get_values("TOTAL SAMPLES", mdas_log_filename))
    mdas_spp_avg = mdas_total_samples / (width * height)

    mdas_initial_sampling_time = sum(get_values("INITIAL SAMPLING TIME", mdas_log_filename))
    mdas_construct_time = sum(get_values("CONSTRUCT TIME", mdas_log_filename))
    mdas_compute_errors_time = sum(get_values("COMPUTE ERRORS TIME", mdas_log_filename))
    mdas_adaptive_sampling_time = sum(get_values("ADAPTIVE SAMPLING TIME", mdas_log_filename))
    mdas_update_indices_time = sum(get_values("UPDATE INDICES TIME", mdas_log_filename))
    mdas_integrate_time = sum(get_values("INTEGRATE TIME", mdas_log_filename))
    mdas_trace_time = sum(get_values("TRACE TIME", mdas_log_filename))

    mdas_total_time = 0
    mdas_total_time += mdas_initial_sampling_time
    mdas_total_time += mdas_construct_time
    mdas_total_time += mdas_compute_errors_time
    mdas_total_time += mdas_adaptive_sampling_time
    mdas_total_time += mdas_update_indices_time
    mdas_total_time += mdas_integrate_time
    mdas_total_time += mdas_trace_time

    mc_trace_time = sum(get_values("TRACE TIME", mc_log_filename))
    mc_total_time = mc_trace_time

    # print time and error
    out_filename = scene + ".tex"
    out_file = open(os.path.join(out_dir, out_filename), "w")
    out_file.write(mdas_test_name + "\n")
    out_file.write(str(p2 % mdas_spp_avg) + " samples per pixel\\\\\n")
    out_file.write("Render time " + str(p0 % mdas_total_time) + " ms\\\\\n")
    out_file.write("MSE " + str(s % mdas_error) + "\n")
    out_file.write("\n")
    out_file.write(mc_test_name + "\n")
    out_file.write(str(mc_spp) + " samples per pixel\\\\\n")
    out_file.write("Render time " + str(p0 % mc_total_time) + " ms\\\\\n")
    out_file.write("MSE " + str(s % mc_error) + "\n")
    out_file.write("\n")

    # hdr to ldr
    ref_image_ldr = 255 * np.clip(ref_image ** (1 / gamma), 0, 1)
    mdas_image_ldr = 255 * np.clip(mdas_image ** (1 / gamma), 0, 1)
    mc_image_ldr = 255 * np.clip(mc_image ** (1 / gamma), 0, 1)

    # crop
    col0 = [35, 51, 239]
    col1 = [70, 187, 95]
    bw = 7
    bwc = 2
    (ref_image_ldr, ref_image_ldr_closeup0) = crop_image(ref_image_ldr, rect0[0][0], rect0[0][1], rect0[1][0], rect0[1][1], bw, bwc, col0)
    (ref_image_ldr, ref_image_ldr_closeup1) = crop_image(ref_image_ldr, rect1[0][0], rect1[0][1], rect1[1][0], rect1[1][1], bw, bwc, col1)
    (mdas_image_ldr, mdas_image_ldr_closeup0) = crop_image(mdas_image_ldr, rect0[0][0], rect0[0][1], rect0[1][0], rect0[1][1], bw, bwc, col0)
    (mdas_image_ldr, mdas_image_ldr_closeup1) = crop_image(mdas_image_ldr, rect1[0][0], rect1[0][1], rect1[1][0], rect1[1][1], bw, bwc, col1)
    (mc_image_ldr, mc_image_ldr_closeup0) = crop_image(mc_image_ldr, rect0[0][0], rect0[0][1], rect0[1][0], rect0[1][1], bw, bwc, col0)
    (mc_image_ldr, mc_image_ldr_closeup1) = crop_image(mc_image_ldr, rect1[0][0], rect1[0][1], rect1[1][0], rect1[1][1], bw, bwc, col1)

    # write highres ldr
    cv2.imwrite(os.path.join(highres_dir, scene + "-ref.png"), ref_image_ldr)
    cv2.imwrite(os.path.join(highres_dir, scene + "-mdas.png"), mdas_image_ldr)
    cv2.imwrite(os.path.join(highres_dir, scene + "-mc.png"), mc_image_ldr)
    cv2.imwrite(os.path.join(highres_dir, scene + "-closeup0-ref.png"), ref_image_ldr_closeup0)
    cv2.imwrite(os.path.join(highres_dir, scene + "-closeup1-ref.png"), ref_image_ldr_closeup1)
    cv2.imwrite(os.path.join(highres_dir, scene + "-closeup0-mdas.png"), mdas_image_ldr_closeup0)
    cv2.imwrite(os.path.join(highres_dir, scene + "-closeup1-mdas.png"), mdas_image_ldr_closeup1)
    cv2.imwrite(os.path.join(highres_dir, scene + "-closeup0-mc.png"), mc_image_ldr_closeup0)
    cv2.imwrite(os.path.join(highres_dir, scene + "-closeup1-mc.png"), mc_image_ldr_closeup1)

    # resize
    max_edge = max(width, height)
    scale = 512 / max_edge
    dim_lowres = (int(width * scale), int(height * scale))
    ref_image_ldr_lowres = cv2.resize(ref_image_ldr, dim_lowres)
    mdas_image_ldr_lowres = cv2.resize(mdas_image_ldr, dim_lowres)
    mc_image_ldr_lowres = cv2.resize(mc_image_ldr, dim_lowres)

    # write lowres ldr
    cv2.imwrite(os.path.join(lowres_dir, scene + "-ref.png"), ref_image_ldr_lowres)
    cv2.imwrite(os.path.join(lowres_dir, scene + "-mdas.png"), mdas_image_ldr_lowres)
    cv2.imwrite(os.path.join(lowres_dir, scene + "-mc.png"), mc_image_ldr_lowres)
    cv2.imwrite(os.path.join(lowres_dir, scene + "-closeup0-ref.png"), ref_image_ldr_closeup0)
    cv2.imwrite(os.path.join(lowres_dir, scene + "-closeup1-ref.png"), ref_image_ldr_closeup1)
    cv2.imwrite(os.path.join(lowres_dir, scene + "-closeup0-mdas.png"), mdas_image_ldr_closeup0)
    cv2.imwrite(os.path.join(lowres_dir, scene + "-closeup1-mdas.png"), mdas_image_ldr_closeup1)
    cv2.imwrite(os.path.join(lowres_dir, scene + "-closeup0-mc.png"), mc_image_ldr_closeup0)
    cv2.imwrite(os.path.join(lowres_dir, scene + "-closeup1-mc.png"), mc_image_ldr_closeup1)

# scene, bin_label, mdas_spp, mc_spp, morton_bit, extra_img_bit, scale_factor, error_threshold, gamma, rect0, rect1
# pool-mb-mdas-mb-1-eib-8-sf-0.0625-et-0.01-spp-8

run("pool", "mb", 8, 14, 1, 8, 0.0625, 0.01, 2.2, [[364, 340], [50, 50]], [[234, 757], [50, 50]])
