import os
import numpy as np

home_drive = "C:/Users/rpr/Desktop/meistdan"
base_dir = home_drive + "/optix/SDK/data/"

test_dir = os.path.join(base_dir, "test-jcgt-rev")

out_dir = os.path.join(base_dir, "jcgt-rev")
if not (os.path.exists(out_dir)):
    os.mkdir(out_dir)

p = "%.2f"


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


def run(index, scene, effect_label, bin_label, tri_num, spp, morton_bit, extra_img_bit, scale_factor,
        alpha, test_dir):

    # test name
    test_name = scene + "-" + bin_label
    test_name += "-mdas"
    test_name += "-mb-" + str(morton_bit)
    test_name += "-eib-" + str(extra_img_bit)
    test_name += "-sf-" + str(scale_factor)
    test_name += "-et-" + str(alpha)
    test_name += "-spp-" + str(spp)
    test_name = test_name + "-pass-0"

    # scene name
    print(test_name)

    # log filename
    log_filename = os.path.join(test_dir, test_name + ".exr.log")

    # parse
    total_samples = sum(get_values("TOTAL SAMPLES", log_filename))
    total_iterations = len(get_values("ADAPTIVE SAMPLING TIME", log_filename))
    initial_sampling_time = sum(get_values("INITIAL SAMPLING TIME", log_filename))
    compute_errors_time = sum(get_values("COMPUTE ERRORS TIME", log_filename))
    adaptive_sampling_time = sum(get_values("ADAPTIVE SAMPLING TIME", log_filename))
    update_indices_time = sum(get_values("UPDATE INDICES TIME", log_filename))
    integrate_time = sum(get_values("INTEGRATE TIME", log_filename))
    trace_time = sum(get_values("TRACE TIME", log_filename))

    integrate_time += update_indices_time

    total_mdas_time = 0
    total_mdas_time += initial_sampling_time
    total_mdas_time += compute_errors_time
    total_mdas_time += adaptive_sampling_time
    total_mdas_time += integrate_time

    total_time = total_mdas_time + trace_time

    table[index][0] = effect_label
    table[index][1] = tri_num
    table[index][2] = str(int(total_samples))
    table[index][3] = str(int(total_iterations))
    table[index][4] = str(p % initial_sampling_time)
    table[index][5] = str(p % compute_errors_time)
    table[index][6] = str(p % adaptive_sampling_time)
    table[index][7] = str(p % integrate_time)
    table[index][8] = str(p % total_mdas_time)
    table[index][9] = str(p % trace_time)
    table[index][10] = str(p % total_time)


# create table
cols = 11
rows= 8
table = [["" for j in range(cols)] for i in range(rows)]

table[0][0] = "Effect"
table[0][1] = "Tiangles"
table[0][2] = "Smples"
table[0][3] = "Iterations"

table[0][4] = "Initial sampl. time [ms]"
table[0][5] = "Error comp. time [ms]"
table[0][6] = "Adaptive sampl. time [ms]"
table[0][7] = "Reconstruction time [ms]"

table[0][8] = "Total sampl. time [ms]"
table[0][9] = "Trace time [ms]"
table[0][10] = "Total time [ms]"

# pool-mb-mdas-mb-1-eib-8-sf-0.0625-et-0.25-spp-4
run(1, "pool", "MB", "mb", "57k", 4, 1, 8, 0.0625, 0.25, test_dir)
# chess-dof-mdas-mb-1-eib-8-sf-1-et-0.25-spp-8
run(2, "chess", "DOF", "dof", "50k", 8, 1, 8, 1, 0.25, test_dir)
# Bistro-dof-mdas-mb-0-eib-10-sf-0.0625-et-0.03125-spp-8
run(3, "Bistro", "DOF", "dof", "3858k", 8, 0, 10, 0.0625, 0.03125, test_dir)
# dragon-dl-mdas-mb-0-eib-10-sf-0.25-et-0.015625-spp-8
run(4, "dragon", "DL", "dl", "871k", 8, 0, 10, 0.25, 0.015625, test_dir)
# cobblestone-dl-mdas-mb-0-eib-10-sf-0.0625-et-0.015625-spp-4
run(5, "cobblestone", "DL", "dl", "10323k", 4, 0, 10, 0.0625, 0.015625, test_dir)
# cornell-box-pt-mdas-mb-1-eib-7-sf-1-et-0.0625-spp-8
run(6, "cornell-box", "PT", "pt", "36", 8, 1, 7, 1, 0.0625, test_dir)
# breakfast-pt-mdas-mb-1-eib-8-sf-0.5-et-0.015625-spp-8
run(7, "breakfast", "PT", "pt", "808k", 8, 1, 8, 0.5, 0.015625, test_dir)

# table file
table_filename = os.path.join(out_dir, "table.tex")
if os.path.exists(table_filename):
    os.remove(table_filename)

# write to file
table_file = open(table_filename, "a")
for j in range(cols):
    for i in range(rows):
        if i > 0:
            table_file.write(" & ")
        table_file.write(table[i][j])
    table_file.write("\\\\\n")
    if j == 3 or j == 7 or j == 9:
        table_file.write("\\hline")
        table_file.write("\n")
table_file.close()
