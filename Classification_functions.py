'''
In order for matlab.engine to work you need to have Matlab installed
and to make sure the package release corresponds to the version of Matlab that is installed on your PC

In my case, I'm using Python 10, and my Matlab version is 2023a, and so the relevant version of
matlab.engine is 9.14.3

You can find the Python versions compatible with Matlab products here:
https://uk.mathworks.com/support/requirements/python-compatibility.html
'''

# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import matlab.engine
from PIL import Image

# Declare constants
block_column = 96
block_row = 96
flatness_criteria = 0.5

# Declare directories
dir_list = 'files_list.txt'
dir_bands = 'C:/Users/mikig/Dropbox/bands_pictures/'
dir_net = 'net.mat'

# Declare how may of the materials you want to go through
start_index = 0
end_index = 10

# Initialize Matlab
eng = matlab.engine.start_matlab()

def read_materials(materials_directory):
    '''
    Read all the material ids from the txt file

    :param materials_directory: str
    :return: list
    '''
    with open(materials_directory, "r") as file:
        # Read the lines from the file
        lines = file.readlines()
        # Remove newline characters and create a list
        materials = [line.strip() for line in lines]
    return materials


def segmentation(bands_directory, begin_ind, end_ind):
    '''
    Segmentation module: Importing and segmenting png files.
                         The file returns a numpy array that contains all the segments of all pictures

    :param bands_directory: str
    :param begin_ind: int
    :param end_ind: int
    :return: np.array, list
    '''
    arr_list = []
    image_segments = []
    images = []
    for x in range(begin_ind, end_ind + 1):
        input_image_path = bands_directory + read_materials(dir_list)[x] + '.png'

        # Open the image and convert it to grayscale
        img = Image.open(input_image_path)
        img = img.convert('L')

        # Images are for the purpose of visualizing the segmented band structures
        image_array = np.array(img)
        images.append(image_array)

        # Get the original width and height
        original_width, original_height = img.size

        # Define the target size (96x96 pixels)
        target_size = (96, 96)

        # Calculate the number of 96x96 segments to extract horizontally
        num_horizontal_segments = original_width // target_size[0]
        num_vertical_segments = original_height // target_size[1]

        # Initialize an empty list for segments
        segments = []

        segment_number = 0
        for i in range(num_horizontal_segments):
            for j in range(num_vertical_segments):
                # Calculate the cropping region for the current segment
                left = i * target_size[0]
                upper = j * target_size[1]
                right = (i + 1) * target_size[0]
                lower = (j + 1) * target_size[1]

                # Crop the segment from the original image
                segment = img.crop((left, upper, right, lower))
                segment_number += 1
                segment_arr = np.array(segment)[:, :, np.newaxis]

                # Append the segment to the list
                segments.append(segment_arr)

                arr_temp = np.zeros(segment_number)

        # Append the arr_temp to arr_list
        arr_list.append(arr_temp)

        # Append the segments to the list
        image_segments.extend(segments)

    image_mat = np.array(image_segments)
    image_mat = np.moveaxis(image_mat, 0, 3)
    image_mat_predictions = np.ascontiguousarray(image_mat)
    return image_mat_predictions, arr_list, images

def predictions(net_directory, array_image_mat):
    '''
    Using the Matlab CNN, assign a flatness score to each segment from every band structure

    :param net_directory: str
    :param array_image_mat: np.array
    :return: np.array
    '''
    # Load Matlab CNN
    model_data = eng.load(net_directory, nargout=1)  # Directory of the loaded network to be changed
    neural_network_model = model_data['net']
    # Predict each segment and save all predictions as a numpy array
    output_data = eng.predict(neural_network_model, array_image_mat, nargout=1)
    predictions = np.array(output_data._data).flatten()

    # For predictions < 0, change them to 0, and for predictions > 1, change them to 1
    for i in range(len(predictions)):
        if predictions[i] < 0:
            predictions[i] = 0
        elif predictions[i] > 1:
            predictions[i] = 1
    return predictions

def assign_predictions(array_list, predictions_array):
    '''
    Start with a list of all numpy arrays, each has shape of the corresponding segmented band structure.
    Assign each segment from the array its corresponding flatness prediction.

    :param array_list: list
    :param predictions_list: np.array
    :return: list
    '''
    current_index = 0
    # The loop iterates over the numpy arrays stored in the array_list
    for i in range(len(array_list)):
        arr_length = len(array_list[i])
        array_list[i][:] = predictions_array[current_index: current_index + arr_length]
        current_index += arr_length

    # Make sure each numpy array from array_list is of correct shape
    for i in range(len(array_list)):
        array_list[i] = array_list[i].reshape((-1, 4))
        array_list[i] = np.transpose(array_list[i])
        array_list[i] = array_list[i].flatten()
    return array_list

def one_flat_segment_predictions(array_mat):
    '''
    Predict how many band structures have at least one flat segment

    :param array_mat: np.array
    :return: 0
    '''
    flat = []
    not_flat = []

    # Flat and non-flat segments get appended to two different lists
    for i in range(len(array_mat)):
        flat_band_present = any(element >= flatness_criteria for element in array_mat[i][:])
        if flat_band_present:
            flat.append(array_mat[i][:])
        else:
            not_flat.append(array_mat[i][:])

    total_flat = len(flat)
    total_not_flat = len(not_flat)
    print(f"Arrays with at least one flat segment: {total_flat}")
    print(f"Arrays without any flat segments: {total_not_flat}")

    # Fraction of band structures that have at least one flat segment
    fraction = (total_flat / (total_flat + total_not_flat)) * 100
    print(f"The fraction of arrays with at least one flat segments: {np.round(fraction)}%")
    print("The number of materials analyzed: ", total_flat + total_not_flat)
    return 0

def flat_pair_predictions(array_mat):
    '''
    Predict how many band structures have at least one pair of flat segments

    :param array_mat: np.array
    :return: 0
    '''
    flat_pairs = []
    not_flat_pairs = []

    for i in range(len(array_mat)):
        pair_found = False

        # Assuming len(arr_list[i]) > 1 to avoid index out of range for j + 1
        for j in range(len(array_mat[i]) - 1):
            # Extracting two neighboring elements
            current_element = array_mat[i][j]
            next_element = array_mat[i][j + 1]

            # Check if both elements are greater than or equal to 0.5
            if current_element >= flatness_criteria and next_element >= flatness_criteria:
                flat_pairs.append(array_mat[i][:])
                pair_found = True
                break  # Break out of the loop once a pair is found

        if not pair_found:
            not_flat_pairs.append(array_mat[i][:])

    total_flat = len(flat_pairs)
    total_not_flat = len(not_flat_pairs)

    print(f"Arrays with at least one pair of neighboring flat segments: {total_flat}")
    print(f"Arrays without any pair of neighboring flat segments: {total_not_flat}")

    # Fraction of band structures with at least one pair of neighboring flat segments
    fraction = (total_flat / (total_flat + total_not_flat)) * 100
    print(f"The fraction of arrays with at least one pair of neighboring flat segments: {np.round(fraction)}%")
    return 0

def flat_row_predictions(array_mat):
    flat_row_id_list = []
    flat_arrays = []
    not_flat_arrays = []

    count = 0
    for arr in array_mat:
        # Reshape the array to have 4 rows and corresponding number of columns
        reshaped_arr = arr.reshape(4, -1)

        # Check if at least one row has all elements >= 0.5
        if any(all(element >= flatness_criteria for element in row) for row in reshaped_arr):
            flat_arrays.append(arr)
            flat_row_id_list.append(read_materials(dir_list)[count + start_index])
            # print(count_0)
        else:
            not_flat_arrays.append(arr)
        count += 1

    total_flat = len(flat_arrays)
    total_not_flat = len(not_flat_arrays)

    print(f"Arrays with at least one fully flat row: {total_flat}")
    print(f"Arrays without any fully flat row: {total_not_flat}")

    # Calculate the fraction
    fraction = (total_flat / (total_flat + total_not_flat)) * 100
    print(f"The fraction of arrays with at least one fully flat rows: {np.round(fraction)}%")
    return flat_row_id_list

def compound_score(array_mat):
    flat_compound_id_list = []
    compound_flatness_score_all = []

    count = 0
    for arr in array_mat:
        # Reshape the array to have 4 rows and the corresponding number of columns
        reshaped_arr = arr.reshape(4, -1)
        # print(np.shape(reshaped_arr)[1])

        # Check if at least one row has all elements >= flatness_score
        flatness_scores_rows = []  # List of flatness scores for all rows in one band structure
        for row in range(len(reshaped_arr)):
            row_flatness_sum = 0

            for element in reshaped_arr[row]:
                row_flatness_sum += element / np.shape(reshaped_arr)[1]  # Average of the flatness scores for each row
            flatness_scores_rows.append(row_flatness_sum)

            compound_flatness_score = max(flatness_scores_rows)  # Compound flatness score for a given material
        if compound_flatness_score >= flatness_criteria:
            flat_compound_id_list.append(read_materials(dir_list)[count + start_index])
            # print(count)
        count += 1

        compound_flatness_score_all.append(compound_flatness_score)

    # print(compound_flatness_score_all)

    flat_materials = []
    not_flat_materials = []
    for element in compound_flatness_score_all:
        if element >= flatness_criteria:
            flat_materials.append(element)
        else:
            not_flat_materials.append(element)

    number_flat_materials = len(flat_materials)
    number_not_flat_materials = len(not_flat_materials)

    print('The number of flat materials is: ', number_flat_materials)
    print('The number of not flat materials is: ', number_not_flat_materials)
    print('The total number of materials is: ', number_flat_materials + number_not_flat_materials)

    fraction = np.round(number_flat_materials / (number_flat_materials + number_not_flat_materials) * 100, 2)
    print(f"The fraction of flat materials is: {fraction}%")
    return flat_compound_id_list, compound_flatness_score_all

def histogram(compound_score_list, start_ind, end_ind):
    # Create a histogram
    plt.hist(compound_score_list, bins=50, color='blue', edgecolor='black')

    title = f'Compound Flatness Score Histogram {start_ind}_to_{end_ind}'
    # Add title and labels
    plt.title(title)
    plt.xlabel('Compound Flatness Score')
    plt.ylabel('Number of materials')

    # Show the plot
    plt.show()

def visualization(array_list, images_array):
    '''
    Visualize each segmented band structure with each segmented having its flatness prediction shown in the image

    :param array_list:
    :param images:
    :param materials_list:
    :return:
    '''
    # Initialize an empty list with the purpose of storing all images
    images_list = []
    for x in range(len(images_array)):
        input_image = images_array[x]
        size_row = int(input_image.shape[0] / block_row)
        size_column = int(input_image.shape[1] / block_column)
        total_blocks = int(size_row * size_column)
        image2_segments = []

        # divided_image = np.zeros((size_row, size_column, block_row, block_column))
        divided_image = np.zeros((total_blocks, block_row, block_column))
        if input_image.shape[0] % block_row == 0 or input_image.shape[1] % block_column == 0:
            for i in range(size_row):
                for j in range(size_column):
                    index = i * size_column + j
                    divided_image[index] = input_image[(i * block_row):((i + 1) * block_row),
                                           (j * block_column):((j + 1) * block_column)]
                    image2_segments.append(divided_image[index])

        images_list.append(image2_segments)

    count = 0
    for element in images_list:
        segments = np.array(element)

        # Calculate the number of rows and columns for the subplot grid
        num_segments = len(segments)
        num_rows = 4  # You can adjust the number of columns as per your preference
        num_cols = -(-num_segments // num_rows)  # Ceiling division to ensure all segments are covered
        # print(num_rows)
        # print(num_cols)

        fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 4 * num_rows))
        axes = axes.flatten()
        # print(num_segments)

        # Flatten the 2D array of subplots for easier indexing
        axes = axes.flatten()

        # Display each image segment in a separate subplot - it works only for a single band structure!!
        for i in range(num_segments):
            extent = [0, 96, 0, 96]  # Extent of the image
            axes[i].imshow(segments[i], cmap='gray', extent=extent)
            axes[i].set_title(f'{array_list[count][i]:.5f}')
            axes[i].axis('off')  # Turn off axis labels
            if array_list[count][i] > flatness_criteria:
                rect = plt.Rectangle((0, 0), 96, 96, edgecolor='red', linewidth=3, fill=False)
            else:
                rect = plt.Rectangle((0, 0), 96, 96, edgecolor='black', linewidth=1, fill=False)
            axes[i].add_patch(rect)
        count += 1
        # Specify where you want to download the plots
        plt.title(f'{read_materials(dir_list)[i]}', loc='center', pad=20, fontsize=16)
        plt.savefig(f'C:/Users/mikig/Dropbox/Classified_bands/{read_materials(dir_list)[i]}.png')
        plt.show()

def save_flat_materials(flat_compound_list, flat_row_list):
    ''''''


    ''''''
    with open('row_flat_ids.txt', 'a') as file_row:
        for element in flat_compound_list:
            file_row.write(element)
            file_row.write('\n')

    with open('compound_flat_ids.txt', 'a') as file_compound:
        for element in flat_row_list:
            file_compound.write(element)
            file_compound.write('\n')

def main():
    '''


    :return:
    '''
    image_mat_list, array_list, images_array = segmentation(dir_bands, start_index, end_index)
    predictions_array = predictions(dir_net, image_mat_list)
    array = assign_predictions(array_list, predictions_array)

    # one_flat_segment_predictions(array)
    # flat_pair_predictions(array)
    flat_row_predictions(array)
    compound_score(array)
    # histogram()
    visualization(array_list, images_array)

main()

# Quit Matlab engine
eng.quit()
