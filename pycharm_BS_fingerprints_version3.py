# import celina

# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from mp_api.client.mprester import MPRester
from pymatgen.electronic_structure.plotter import BSPlotter
from pymatgen.util.plotting import add_fig_kwargs, get_ax3d_fig, pretty_plot


MAPI_KEY = "bbiUJxPj5bmp31SqSsuc0EAcO3ekNw6f"  # You must change this to your Materials API key! (or set MAPI_KEY env variable)
mpr = MPRester(MAPI_KEY)  # object for connecting to MP Rest interface
number_points = 20
directory = 'sorted_files_list.txt'

def read_materials(materials_directory):
    with open(materials_directory, 'r') as file:
        lines = file.readlines()
        materials = [line.strip() for line in lines]
    return materials

# def dos_download(mpr):
#     for id_ in range(1, 2):
#         mp_id = "mp-" + str(id_)  # You must change this to the mp-id of your compound of interest
#         # try:
#         bs = mpr.get_bandstructure_by_material_id(mp_id)
#
#             # BSPlotter(bs).show()
#         my_bs = bs.as_dict()
#             # print(my_bs.keys())
#             # print(my_bs['band_gap'])
#             # print(my_bs['lattice_rec'])
#             # print(my_bs['kpoints'])
#             # print(len(my_bs['kpoints']))
#             # print(my_bs['projections'])
#         print(my_bs['kpoints'])
#         kpoints = my_bs['kpoints']
#
#
#         # except:
#         #     print('Band not downloaded')
#         #     pass
#
#
# def main():
#     dos_download(mpr)


materials = read_materials(directory)
for material in materials[46:47]:
    # mp_id = "mp-" + str(id_)  # You must change this to the mp-id of your compound of interest
    print(material)
    # try:
    bs = mpr.get_bandstructure_by_material_id(material)
    # structure = mpr.get_structure_by_material_id(mp_id)

    plot = BSPlotter(bs)
    plot.show()
    my_bs = bs.as_dict()
    # print(my_bs.keys())
    # print(my_bs['band_gap'])
    # print(my_bs['lattice_rec'])
    # print(my_bs['kpoints'])
    # print(len(my_bs['kpoints']))
    # print(my_bs['projections'])
    # print(my_bs['kpoints'])
    kpoints = my_bs['kpoints']

    # print(my_bs['bands'])
    bands = my_bs['bands']
    # print(bands.keys())
    list = bands['1']
    array = np.array(list)
    # print(array)
    data = BSPlotter(bs).bs_plot_data()
    # print(data)
    distances = data['distances']
    distances2 = data['distances']
    energies = data["energy"]['1']
    energies2 = data["energy"]['1']
    # print(len(energy))
    data_plot = np.array(my_bs['bands']['1']) - my_bs['efermi']
    # BSPlotterProjected(bs).get_projected_plots_dots()
    # print(array[0])
    # print(kpoints)
    # print(data_plot.shape)


# structure = bs.structure
# cart = structure.cart_coords
# kpath = HighSymmKpath(structure)
# labels = kpath._get_klabels()
# band = HighSymmKpath.get_continuous_path(bs)

# print(kpath.kpath['kpoints'].values())
# kpath_dict = kpath.kpath['kpoints']
# kpoints_list = kpath_dict.values()
# kpoints_labels = kpath_dict.keys()

# length = kpath.path_lengths()

# for k, label in zip(kpoints, labels):
#     print(label, k)

ax = pretty_plot(12, 8)

intervals = [0]
list_intervals = [0]
for i in range(len(distances)):
    print(distances[i].shape)
    # high_symm_pts.append()
    intervals.append(0 + distances[i].shape[0])
    print(intervals[:i+2])
    list_intervals.append(sum(intervals[:i+2]))

# high_symm_pts = []
# for i in list_intervals:
#     if i == 0:
#         high_symm_pts.append(kpoints[i])
#     else:
#         high_symm_pts.append(kpoints[i - 1])

high_symm_pts = []
for i in list_intervals:
    if i == 0:
        high_symm_pts.append(kpoints[i])
    else:
        high_symm_pts.append(kpoints[i - 1])



for i in range(len(energies)):
    # print(energies[i][0, :].shape)
    plt.plot(distances[i], energies[i][0, :], color='black', linewidth=5)
    plt.ylim(-15, 15)


plt.show()



data_x = []
data_y = []
for i in range(len(distances)):
    print(i)
    data_x.append(distances[i])
    data_y.append(energies[i][0, :])

# print(data_x.shape)

xvals = np.linspace(0, max(data_x[0]), 20)
y_interp = np.interp(xvals, data_x[0], data_y[0])


def interpolation(data_x, data_y):
    x_vals = np.linspace(min(data_x), max(data_x), number_points)
    y_vals = np.interp(x_vals, data_x, data_y)
    return x_vals, y_vals

data_x_interpolated = []
data_y_interpolated = []

for i in range(len(data_x)):
    plt.plot(interpolation(data_x[i], data_y[i])[0], interpolation(data_x[i], data_y[i])[1], number_points,
             color='black', linewidth=3)
    data_x_interpolated.append(interpolation(data_x[i], data_y[i])[0])
    data_y_interpolated.append(interpolation(data_x[i], data_y[i])[1])

intervals = [0]
list_intervals = [0]
for i in range(len(distances)):
    print(distances[i].shape)
    # high_symm_pts.append()
    intervals.append(0 + distances[i].shape[0])
    print(intervals[:i+2])
    list_intervals.append(sum(intervals[:i+2]))

plt.ylim(-50, 50)
plt.title('ESSA')
plt.show()

distances = np.hstack(distances)
energies = np.hstack(energies)
steps = BSPlotter._get_branch_steps(bs.branches)[10:-10]
distances = np.split(distances, steps)
energies = np.hsplit(energies, steps)


count = 0
energy_min = -1000
energy_max = 1000
list_energies = []
for dist, ene in zip(distances, energies):
    print(len(energies))
    for i in range(ene.T.shape[1]):
        list_temp = []
        for j in ene.T[:, i]:
            if any(j >= energy_min and j <= energy_max for j in ene.T[:, i]):
                list_temp.append(j)
        list_energies.append(list_temp)
    print('essa=', len(list_energies))

# count_bands = 0
# list_energies_2 = []
# for dist, ene in zip(distances, energies):
#     for i in range(ene.T.shape[1]):
#         list_temp = []
#         energy_min = -5
#         energy_max = 5
#         for j in ene.T[:, i]:
#             if any(j >= energy_min and j <= energy_max for j in ene.T[:, i]):
#                 list_temp.append(j)
#         count_bands += 1
#         if count_bands == 6:
#             break
#         list_energies_2.append(list_temp)

number_of_bands = 10


# list_energies = [i for i in list_energies if len(i) != 0]

list_difference = []
for i in range(len(list_energies)):
    max_val = max(list_energies[i])
    min_val = min(list_energies[i])
    difference = abs(max_val - min_val)
    list_difference.append(difference)
print(list_difference)

# number_bands = len(list_energies)

# Want to take 10 bands, 5 below the Fermi energy and 5 above !!!
# The algorithm:
# 1. Take the band whose gamma point element is closest to the Fermi energy
# -> This gives the first band
# 2. Then take 9 bands whose gamma points are closest to the first band
# -> This gives the other 9 bands, resulting in 10 bands in total

# 1. How to implement 1. ?
# I need to take all bands and extract the gamma points for all of them !!!

number_bands = len(list_energies)

gamma_pts = []
for list in list_energies:
    gamma_pts.append(list[0])
    # print(list[0])

min_gamma_pts = min(gamma_pts, key=lambda x:abs(x-0)) # Gamma point closest to the Fermi energy

# Okay, so now I have the gamma point closest to the Fermi energy.
# In case, there is one more point, I also extract it

list_min_pts = []
for pts in gamma_pts:
    if pts == min_gamma_pts:
        list_min_pts.append(pts)
    else:
        continue

# Now I can find 9 points closest to selected point(s)
# Do it recursively

# 0. Load the list with all points minus the og min points -> done
# 1. First, check if the point is not the original min point
# 2. Then, append the list with the min point with respect to the og min point
# 3. Then, take away the new min points from the list
# 4. Finally, call the function again
# 5. Break when there are 9 points found
# Some points are degenerate -> localize them!

gamma_pts = [pts for pts in gamma_pts if pts != min_gamma_pts]

other_pts = []
def find_pts(gamma_pts, other_pts, min_pts):
    while(len(other_pts) < 10):
        min_point = min(gamma_pts, key=lambda x:abs(x-min_pts))
        # print(min_point)
        for pts in gamma_pts:
            if pts == min_point:
                # print(pts)
                other_pts.append(pts)
            else:
                continue
        gamma_pts = [i for i in gamma_pts if i != min_point]
        print(gamma_pts)
        find_pts(gamma_pts, other_pts, min_point)
    return other_pts

other_points = find_pts(gamma_pts, other_pts, min_gamma_pts)

combined_pts = list_min_pts + other_points
combined_pts = combined_pts[:10]
# Combined_pts is the list of all gamma points of the bands that we want to consider
# Now I need to extract the rest of the bands for these gamma points

bands_energies = []

for list in list_energies:
    if list[0] in combined_pts:
        bands_energies.append(list)
    else:
        continue
bands_energies = bands_energies[:10] # 10 is correct here

# bands_energies is the list that contains the energies for 10 bands

bands_distances_pre_pre = []
for array in distances2:
    bands_distances_pre_pre.append(array.tolist())

bands_distances_pre = [item for sublist in bands_distances_pre_pre for item in sublist]

bands_distances = []
for i in range(10):
    bands_distances.append(bands_distances_pre)

bands_new = []
for list in bands_distances:
    for i in range(len(list_intervals) - 1):
        bands_new.append(np.array(list[list_intervals[i]:list_intervals[i+1]]))

compressed_bands_distances = []
for i in range(0, len(bands_new), len(distances2)):
    compressed_bands_distances.append(bands_new[i:i+len(distances2)]) # i+10

bands_ene = []
for list in bands_energies:
    for i in range(len(list_intervals) - 1):
        bands_ene.append(np.array(list[list_intervals[i]:list_intervals[i+1]]))

compressed_bands_energies = []
for i in range(0, len(bands_ene), len(distances2)):
    compressed_bands_energies.append(bands_ene[i:i+len(distances2)]) # i+10

# 1. Load bands_distance
# 2. Each sublist needs to be converted into 10 arrays, each array is of varying length
# 3. The output should be a list of lists of arrays!!!

###################################################
distances_list = dist.tolist()

high_symm_pts_dist = []
for i in list_intervals:
    if i == 0:
        high_symm_pts_dist.append(distances_list[i])
    else:
        high_symm_pts_dist.append(distances_list[i - 1])
#
# data_y_interpolated_list = []
# for i in range(len(data_y_interpolated)):
#     data_y_interpolated_list.append(data_y_interpolated[i].tolist())
#
# interpolated_y_flattened = [item for sublist in data_y_interpolated_list for item in sublist]
#
#
# bs_fingerprints_kpoints = []
#
# interpolated_kpoints = [item for sublist in high_symm_pts for item in sublist]
#
# bs_fingerprints_kpoints.append(interpolated_kpoints)
# bs_fingerprints_kpoints.append(interpolated_y_flattened)
#
# flattened_fingerprint_kpoints = [item for sublist in bs_fingerprints_kpoints for item in sublist]
#
#
# bs_fingerprints_dist = []
#
#
# bs_fingerprints_dist.append(high_symm_pts_dist)
# bs_fingerprints_dist.append(interpolated_y_flattened)
#
# # This is the fingerprint that is fully correct and that can be used as an input for the VAE
# flattened_fingerprint_dist = [item for sublist in bs_fingerprints_dist for item in sublist]


########################################################################################################################
data_x_all = []
data_y_all = []
for i in range(number_of_bands):
    for j in range(len(distances2)):
        data_x_all.append(distances2[j])
        data_y_all.append(energies2[j][i, :])

compressed_x_data = []
compressed_y_data = []
for i in range(0, len(data_x_all), len(distances2)):
    compressed_x_data.append(data_x_all[i:i+len(distances2)]) # i + 10
    compressed_y_data.append(data_y_all[i:i+len(distances2)]) # i + 10

compressed_x_data = compressed_bands_distances
compressed_y_data = compressed_bands_energies

data_x_interpolated_all = []
data_y_interpolated_all = []
for i in range(number_of_bands):
    for j in range(len(distances2)):
        data_x_interpolated_all.append(interpolation(compressed_x_data[i][j], compressed_y_data[i][j])[0])
        data_y_interpolated_all.append(interpolation(compressed_x_data[i][j], compressed_y_data[i][j])[1])


compressed_x_interpolated = []
compressed_y_interpolated = []
for i in range(0, len(data_x_all), len(distances2)):
    compressed_x_interpolated.append(data_x_interpolated_all[i:i+len(distances2)]) # i+10
    compressed_y_interpolated.append(data_y_interpolated_all[i:i+len(distances2)]) # i+10

for i in range(len(distances2)):
    plt.plot(data_x_interpolated_all[i], data_y_interpolated_all[i])

plt.ylim(-15, 15)
plt.title('LOL222')
plt.show()

list_intervals_all = []
for i in range(number_of_bands):
    list_intervals_all.append(list_intervals) # good


high_symm_pts_dist_all = []
for i in range(number_of_bands):
    high_symm_pts_dist_all.append(high_symm_pts_dist) # good

# flattened_high_symmetry = [item for sublist in high_symm_pts_dist_all for item in sublist]

compressed_y_interpolated_list = []
for i in range(number_of_bands):
    for array in compressed_y_interpolated[i]:
        compressed_y_interpolated_list.append(array.tolist())

compressed_y_interpolated_list_list = []
for i in range(0, len(data_x_all), len(distances2)):
    compressed_y_interpolated_list_list.append(compressed_y_interpolated_list[i:i+len(distances2)])


compressed_y_interpolated_list_list_2 = []

bs_fingerprints_dist_list_all = []
flattened_fingerprint = []
for i in range(number_of_bands):
    bs_fingerprints_dist_list_all.append(high_symm_pts_dist_all[i])
    bs_fingerprints_dist_list_all.append(compressed_y_interpolated_list_list[i])

combined_fingerprint = []
for i in range(0, len(bs_fingerprints_dist_list_all), 2):
    combined_fingerprint.append(bs_fingerprints_dist_list_all[i:i+2])

flattened_fingerprint = []
for i in range(number_of_bands):
    flattened_fingerprint = [item for sublist in bs_fingerprints_dist_list_all for item in sublist]

def flatten_list(list):
    flattened_list = []
    for element in list:
        if isinstance(element, type([])):
            flattened_list.extend(flatten_list(element))
        else:
            flattened_list.append(element)
    return flattened_list

almost_final_fingerprint = flatten_list(flattened_fingerprint)

final_fingerprint_list = []
for i in range(0, len(almost_final_fingerprint), len(high_symm_pts_dist) + len(distances2) * number_points):
    final_fingerprint_list.append(almost_final_fingerprint[i:i+len(high_symm_pts_dist) + len(distances2) * number_points])

final_fingerprint_array = np.array(final_fingerprint_list)
array_transposed = final_fingerprint_array.transpose()

final_fingerprint_array = array_transposed
########################################################################################################################
# Now do it for all band structures

# Input the band structures -> ??
# Create a list of these inputs and then select material ids from this list
# This will prevent the API over++++++++++load

# Then just do the previous things but just for all materials from the list
# So in the end I should get a 3D array with data for all materials of interest
# def read_materials(materials_directory):
#     with open(materials_directory, 'r') as file:
#         lines = file.readlines()
#         materials = [line.strip() for line in lines]
#     return materials
#
# directory = 'sorted_files_list.txt'
# materials = read_materials(directory)
#
# for material in materials:
#     bs = mpr.get_bandstructure_by_material_id(material)
#     plot = BSPlotter(bs)
#     plot.show()
#     my_bs = bs.as_dict()
#     kpoints = my_bs['kpoints']
#     bands = my_bs['bands']
#     list = bands['1']
#     array = np.array(list)
#     data = BSPlotter(bs).bs_plot_data()
#     distances = data['distances']
#     distances2 = data['distances']
#     energies = data["energy"]['1']
#     energies2 = data["energy"]['1']









########################################################################################################################
### Here, do the fingerprints from the paper -> 32 bins per special high-symmetry point
# Take the energy range to be (-10, 10) eV and limit the analysis to the Gamma point
# First, divide the region into 10 equal intervals along the energy axis
# Then, look if a given band is in the energy range, i.e. it its energies are within certain
# limits.
#

# energy_range = np.linspace(-10, 10, 11) # This is the energy range
# count = []

# for list in list_energies:
#     element_found = False
#     for i in list:
#         if i >= -10 and i <= -6:
#             count.append(list)
#             element_found = True
#             break

# for energy_range in energy_range:
#     if energy_range < 10:
#         for list in list_energies:
#             element_found = False
#                 # print(range)
#             if list[0] >= energy_range and list[0] <= energy_range + 2:
#                 # print(energy_range)
#                 count.append(list)
#                 element_found = True
#                 break

# number = len(count)

# for range in energy_range:
#     if range < 10:
#         print(range)
#         print(range+2)

# How to get information about a single band structure -> not only its energies but also kpoints and other things
# Each band has the same high-symmtery points and distances, but each has different energies.
# Number of points in a line = length of the distances list

### Now write functions doing the same thing

# def bs_download(mpr):
#     for id_ in range(1, 2):
#         mp_id = "mp-" + str(id_)  # You must change this to the mp-id of your compound of interest
#         try:
#             bs = mpr.get_bandstructure_by_material_id(mp_id)
#
#             # BSPlotter(bs).show()
#             data = BSPlotter(bs).bs_plot_data()
#             # print(data)
#             distances = data['distances']
#             energies = data['energy']['1']
#             return distances, energies
#
#         except:
#             print('Band not downloaded')
#             pass
#
#
# def single_band(distances, energies):
#     ax = pretty_plot(12, 8)
#
#     distances = np.hstack(distances)
#     energies = np.hstack(energies)
#     steps = BSPlotter._get_branch_steps(bs.branches)[1:-1]
#     distances = np.split(distances, steps)
#     energies = np.hsplit(energies, steps)
#
#     for dist, ene in zip(distances, energies):
#         # ax.plot(dist, ene.T, color='black')
#         ax.plot(dist, ene.T[:, 5], color='black', linewidth=2)
#         print(ene.T[:, 0].shape)
#
#     ax.set_ylim(-5, 9)
#     ax.set_xlabel('K-vector')
#     ax.set_ylabel('Energy [eV]')
#     plt.show()
#
# def main():
#     distances, energies = bs_download(mpr)
#     single_band(distances, energies)
#
# main()


## Now do the thing with Anupam's fringerprints
# So the plan is to have a vector consisting of the real k-coordinates of the high-symmetry
# points. Then, in between these vectors have the eigenvalues, let's say 20 of them for each
# interval.
# Things to do: 1. Get coordinates of the high-symmetry points
# 2. Get energies of the k-points -> how to normalize them?

# print(len(list_energies[0])) # This is the list of eigenvalues for one band
# Need to get it for each separate interval


# List of eigenvalues for each interval
# distances_kpoints = []
# for array in distances2:
#     distances_kpoints.append(array.tolist())
#     print(len(array.tolist()))



# plt.plot(distances[0], list_energies[1])
# plt.ylim(-15, 15)
# plt.show()



# print(len(distances[0]), len(list_energies[0]))

# for array in list_energies:
#     print(len(array))

# Now I need to get the k coordinates for each high-symmetry point

