import numpy as np
from mp_api.client.mprester import MPRester
from pymatgen.electronic_structure.plotter import BSPlotter

MAPI_KEY = "bbiUJxPj5bmp31SqSsuc0EAcO3ekNw6f"  # You must change this to your Materials API key! (or set MAPI_KEY env variable)
mpr = MPRester(MAPI_KEY)  # object for connecting to MP Rest interface
number_of_bands = 10
number_of_interpolated_points = 20
directory = 'sorted_files_list.txt'

def get_material_info(mpr, material):
    '''
    Get BS data using this function
    :param mpr:
    :return:
    '''
    print(material)
    bs = mpr.get_bandstructure_by_material_id(material)
    # plot = BSPlotter(bs)
    # plot.show()
    my_bs = bs.as_dict()
    kpoints = my_bs['kpoints']
    bands = my_bs['bands']
    list = bands['1']
    data = BSPlotter(bs).bs_plot_data()
    distances = data['distances']
    energies = data["energy"]['1']
    return bs, distances, energies

def get_intervals(distances):
    '''
    The intervals measure how many points there are between each two high-symmetry points
    :param distances:
    :return:
    '''
    intervals = [0]
    list_intervals = [0]
    for i in range(len(distances)):
        intervals.append(0 + distances[i].shape[0])
        list_intervals.append(sum(intervals[:i + 2]))
    return list_intervals

# def combine_data(bs, distances, energies):
#     '''
#     Use this function to stack the distances and energies from each band together so that there are multiple bands
#     created.
#     :param distances:
#     :param energies:
#     :return:
#     '''
#     distances = np.hstack(distances)
#     energies = np.hstack(energies)
#     steps = BSPlotter._get_branch_steps(bs.branches)[10:-10]
#     distances = np.split(distances, steps)
#     energies = np.hsplit(energies, steps)
#     return distances, energies

def get_energies_list(material):
    '''
    Function to create a list that contains all the energy values for all bands
    :param distances:
    :param energies:
    :return:
    '''
    bs = mpr.get_bandstructure_by_material_id(material)
    # plot = BSPlotter(bs)
    # plot.show()
    my_bs = bs.as_dict()
    kpoints = my_bs['kpoints']
    bands = my_bs['bands']
    list = bands['1']
    data = BSPlotter(bs).bs_plot_data()
    distances = data['distances']
    energies = data["energy"]['1']

    distances = np.hstack(distances)
    energies = np.hstack(energies)
    steps = BSPlotter._get_branch_steps(bs.branches)[10:-10]
    distances = np.split(distances, steps)
    energies = np.hsplit(energies, steps)

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
    return list_energies, dist

def get_pts(gamma_pts, other_pts, min_pts):
    '''
    Function that is used to find points that are closest to the Fermi energy
    :param gamma_pts:
    :param other_pts:
    :param min_pts:
    :return:
    '''
    while(len(other_pts) < 10):
        min_point = min(gamma_pts, key=lambda x:abs(x-min_pts))
        for pts in gamma_pts:
            if pts == min_point:
                other_pts.append(pts)
            else:
                continue
        gamma_pts = [i for i in gamma_pts if i != min_point]
        get_pts(gamma_pts, other_pts, min_point)
    return other_pts

def get_bands(list_energies):
    '''
    Function tha finds the gamma points of those 10 bands that are closest to the Fermi energy -> these bands are then
    used for the fingerprints
    :param list_energies:
    :return:
    '''

    gamma_pts = []
    for list in list_energies:
        gamma_pts.append(list[0])
        # print(list[0])

    min_gamma_pts = min(gamma_pts, key=lambda x: abs(x - 0))  # Gamma point closest to the Fermi energy

    list_min_pts = []
    for pts in gamma_pts:
        if pts == min_gamma_pts:
            list_min_pts.append(pts)
        else:
            continue

    gamma_pts = [pts for pts in gamma_pts if pts != min_gamma_pts]
    other_pts = []
    other_points = get_pts(gamma_pts, other_pts, min_gamma_pts)

    combined_pts = list_min_pts + other_points
    combined_pts = combined_pts[:10]
    return combined_pts

def get_compressed_data(distances2, list_intervals, list_energies, combined_pts):
    '''
    This function is a bit messy but it returns the data for all the x and y coordinates of all 10 bands.
    :param distances2:
    :param list_intervals:
    :param list_energies:
    :param combined_pts:
    :return:
    '''
    bands_energies = []

    for list in list_energies:
        if list[0] in combined_pts:
            bands_energies.append(list)
        else:
            continue
    bands_energies = bands_energies[:10]

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
            bands_new.append(np.array(list[list_intervals[i]:list_intervals[i + 1]]))

    compressed_bands_distances = []
    for i in range(0, len(bands_new), len(distances2)):
        compressed_bands_distances.append(bands_new[i:i + len(distances2)])

    bands_ene = []
    for list in bands_energies:
        for i in range(len(list_intervals) - 1):
            bands_ene.append(np.array(list[list_intervals[i]:list_intervals[i + 1]]))

    compressed_bands_energies = []
    for i in range(0, len(bands_ene), len(distances2)):
        compressed_bands_energies.append(bands_ene[i:i + len(distances2)])
    return compressed_bands_distances, compressed_bands_energies

def data_x_y(distances2, energies2):
    data_x_all = []
    data_y_all = []
    for i in range(number_of_bands):
        for j in range(len(distances2)):
            data_x_all.append(distances2[j])
            data_y_all.append(energies2[j][i, :])
    return data_x_all, data_y_all
def interpolation(data_x, data_y):
    '''
    Function used to interpolate data -> 20 points for each interval by default
    :param data_x:
    :param data_y:
    :return:
    '''
    x_vals = np.linspace(min(data_x), max(data_x), number_of_interpolated_points)
    y_vals = np.interp(x_vals, data_x, data_y)
    return x_vals, y_vals

def interpolate_data(distances2, compressed_x_data, compressed_y_data):
    data_x_interpolated_all = []
    data_y_interpolated_all = []
    for i in range(number_of_bands):
        for j in range(len(distances2)):
            data_x_interpolated_all.append(interpolation(compressed_x_data[i][j], compressed_y_data[i][j])[0])
            data_y_interpolated_all.append(interpolation(compressed_x_data[i][j], compressed_y_data[i][j])[1])
    return data_x_interpolated_all, data_y_interpolated_all

def compress_interpolated_data(data_x_all, distances2, data_x_interpolated_all, data_y_interpolated_all):
    compressed_x_interpolated = []
    compressed_y_interpolated = []
    for i in range(0, len(data_x_all), len(distances2)):
        compressed_x_interpolated.append(data_x_interpolated_all[i:i + len(distances2)])
        compressed_y_interpolated.append(data_y_interpolated_all[i:i + len(distances2)])
    return compressed_x_interpolated, compressed_y_interpolated

def get_high_symm_dist(dist, list_intervals):
    distances_list = dist.tolist()
    high_symm_pts_dist = []
    for i in list_intervals:
        if i == 0:
            high_symm_pts_dist.append(distances_list[i])
        else:
            high_symm_pts_dist.append(distances_list[i - 1])
    return high_symm_pts_dist

def get_intervals_distances(list_intervals, high_symm_pts_dist):
    list_intervals_all = []
    for i in range(number_of_bands):
        list_intervals_all.append(list_intervals)

    high_symm_pts_dist_all = []
    for i in range(number_of_bands):
        high_symm_pts_dist_all.append(high_symm_pts_dist)
    return list_intervals_all, high_symm_pts_dist_all

def get_final_data(compressed_y_interpolated, data_x_all, distances2, high_symm_pts_dist_all):
    compressed_y_interpolated_list = []
    for i in range(number_of_bands):
        for array in compressed_y_interpolated[i]:
            compressed_y_interpolated_list.append(array.tolist())

    compressed_y_interpolated_list_list = []
    for i in range(0, len(data_x_all), len(distances2)):
        compressed_y_interpolated_list_list.append(compressed_y_interpolated_list[i:i + len(distances2)])

    bs_fingerprints_dist_list_all = []
    for i in range(number_of_bands):
        bs_fingerprints_dist_list_all.append(high_symm_pts_dist_all[i])
        bs_fingerprints_dist_list_all.append(compressed_y_interpolated_list_list[i])

    combined_fingerprint = []
    for i in range(0, len(bs_fingerprints_dist_list_all), 2):
        combined_fingerprint.append(bs_fingerprints_dist_list_all[i:i + 2])

    flattened_fingerprint = []
    for i in range(number_of_bands):
        flattened_fingerprint = [item for sublist in bs_fingerprints_dist_list_all for item in sublist]
    return flattened_fingerprint

def flatten_list(list):
    flattened_list = []
    for element in list:
        if isinstance(element, type([])):
            flattened_list.extend(flatten_list(element))
        else:
            flattened_list.append(element)
    return flattened_list

def fingerprint_creation(distances2, high_symm_pts_dist, flattened_fingerprint):
    '''
    Final function -> creates the fingerprint for the entire BS as an array and as a list
    :return:
    '''
    almost_final_fingerprint = flatten_list(flattened_fingerprint)

    final_fingerprint_list = []
    for i in range(0, len(almost_final_fingerprint), len(high_symm_pts_dist) + len(distances2) * number_of_interpolated_points):
        final_fingerprint_list.append(almost_final_fingerprint[i:i + len(high_symm_pts_dist) + len(distances2) * number_of_interpolated_points])

    final_fingerprint_array = np.array(final_fingerprint_list)
    array_transposed = final_fingerprint_array.transpose()

    final_fingerprint_array = array_transposed
    return final_fingerprint_array


def read_materials(materials_directory):
    with open(materials_directory, 'r') as file:
        lines = file.readlines()
        materials = [line.strip() for line in lines]
    return materials


def get_fingerprint_BS(material):
    bs, distances_2, energies_2 = get_material_info(mpr, material)
    list_intervals = get_intervals(distances_2)
    # distances_new, energies_new = combine_data(bs, distances_2, energies_2)
    list_energies, dist = get_energies_list(material)
    if len(list_energies) > 15:
        combined_pts = get_bands(list_energies)
        compressed_x_data, compressed_y_data = get_compressed_data(distances_2, list_intervals, list_energies, combined_pts)
        data_x_all, data_y_all = data_x_y(distances_2, energies_2)
        data_x_interpolated, data_y_interpolated = interpolate_data(distances_2, compressed_x_data, compressed_y_data)
        compressed_x_interpolated, compressed_y_interpolated = compress_interpolated_data(data_x_all, distances_2, data_x_interpolated, data_y_interpolated)
        high_symm_pts_dist = get_high_symm_dist(dist, list_intervals)
        list_intervals_all, high_symm_pts_dist_all = get_intervals_distances(list_intervals, high_symm_pts_dist)
        flattened_fingerprint = get_final_data(compressed_y_interpolated, data_x_all, distances_2, high_symm_pts_dist_all)
        final_fingerprint_array = fingerprint_creation(distances_2, high_symm_pts_dist, flattened_fingerprint)
        return final_fingerprint_array
    else:
        final_fingerprint_array = []
        final_fingerprint_array = np.array(final_fingerprint_array)
        return final_fingerprint_array

def main():
    fingerprint_total = [] # for now save it as a list
    materials = read_materials(directory)
    for material in materials[0:100]: # <- can change the number of materials HERE
        fingerprint_total.append(get_fingerprint_BS(material))
    # fingerprint_total = np.array(fingerprint_total) -> some issues with this line
    return fingerprint_total

fingerprint = main()
