#!/usr/bin/env python
# coding: utf-8

# In[17]:


import numpy as np
import os
#import cv2
#import tensorflow as tf
import matplotlib.pyplot as plt
import matlab.engine
#import matlab
from PIL import Image


# In[18]:


block_column = 96
block_row = 96
start_index = 1
end_index = 3
flatness_criteria = 0.5


# In[19]:


eng = matlab.engine.start_matlab() # Start Matlab engine


# In[ ]:


## Segmentation module
# First, initialize an empty array where the third dim is the colour number (basically representing the .convert('L') function),
# and the last dim is the total number of segments that is then appended in the for loop

## This code does not save the divided images to avoid storing thousands of them in one folder

# Initialize an empty array for image segments
#image_mat = np.empty((96, 96, 1, 0), dtype=np.uint8)
image_mat = np.empty((96, 96, 1), dtype=np.uint8)


arr_list = []
images = []

materials = []
with open("C:/Users/v93777mg/missing_files.txt", "r") as file:
    # Read the lines from the file
    lines = file.readlines()

# Remove newline characters and create a list
materials = [line.strip() for line in lines]

image_segments = []
for x in range(0, 2500):  # Assuming a fixed range for x, adjust as needed
    directory = f'C:/Users/v93777mg/bands_pictures/'  # Directory of the full images to be changed
    input_image_path = directory + materials[x] + '.png'
    print(x)

    img = Image.open(input_image_path)

    # Convert the image to grayscale
    img = img.convert('L')

    # Get the original width and height
    original_width, original_height = img.size

    # Define the target size (96x96 pixels)
    target_size = (96, 96)

    # Calculate the number of 96x96 segments to extract horizontally
    num_horizontal_segments = original_width // target_size[0]
    num_vertical_segments = original_height // target_size[1]

    segment_number = 0
    # Initialize an empty list for segments
    segments = []

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

# Convert the list of segments to a numpy array
image_mat2 = np.array(image_segments)
image_mat3 = np.moveaxis(image_mat2, 0, 3)
image_mat_contiguous = np.ascontiguousarray(image_mat3)


# for x in range(0, 100):#len(materials)):
#     directory = f'C:/Users/v93777mg/bands_pictures/'  # Directory of the full images to be changed
#     # band_name = "mp-" + str(x) + ".png"
#     # input_image_path = directory + band_name
#     input_image_path = directory + materials[x] + '.png'
#     print(x)
#     #
#     # if not os.path.isfile(input_image_path):
#     #     print("No such file %s" % input_image_path)
#     #     continue
#
#     img = Image.open(input_image_path)
#
#     # Convert the image to grayscale
#     img = img.convert('L')
#
#     # image_array = np.array(img)
#     # images.append(image_array)
#
#     # Get the original width and height
#     original_width, original_height = img.size
#
#     # Define the target size (96x96 pixels)
#     target_size = (96, 96)
#
#     # Calculate the number of 96x96 segments to extract horizontally
#     num_horizontal_segments = original_width // target_size[0]
#     num_vertical_segments = original_height // target_size[1]
#
#     segment_number = 0
#     for i in range(num_horizontal_segments):
#         for j in range(num_vertical_segments):
#             # Calculate the cropping region for the current segment
#             left = i * target_size[0]
#             upper = j * target_size[1]
#             right = (i + 1) * target_size[0]
#             lower = (j + 1) * target_size[1]
#
#             # Crop the segment from the original image
#             segment = img.crop((left, upper, right, lower))
#             segment_number += 1
#             segment_arr = np.array(segment)[:, :, np.newaxis, np.newaxis]
#             image_mat = np.concatenate((image_mat, segment_arr),
#                                        axis=-1)  # This concatenation is for the puroposes of the Matlab .predict function
#
#             arr_temp = np.zeros(segment_number)
#     arr_list.append(arr_temp)

# for x in range(start_index, end_index + 1):
#         directory = f'C:/Users/v93777mg/bands_pictures/' # Directory of the full images to be changed
#         band_name = "mp-" + str(x) + ".png"
#         input_image_path = directory + band_name
#
#         if not os.path.isfile(input_image_path):
#             print("No such file %s" % input_image_path)
#             continue
#
#
#         img = Image.open(input_image_path)
#
#         # Convert the image to grayscale
#         img = img.convert('L')
#
#         #image_array = np.array(img)
#         #images.append(image_array)
#
#         # Get the original width and height
#         original_width, original_height = img.size
#
#         # Define the target size (96x96 pixels)
#         target_size = (96, 96)
#
#         # Calculate the number of 96x96 segments to extract horizontally
#         num_horizontal_segments = original_width // target_size[0]
#         num_vertical_segments = original_height // target_size[1]
#
#         segment_number = 0
#         for i in range(num_horizontal_segments):
#             for j in range(num_vertical_segments):
#                 # Calculate the cropping region for the current segment
#                 left = i * target_size[0]
#                 upper = j * target_size[1]
#                 right = (i + 1) * target_size[0]
#                 lower = (j + 1) * target_size[1]
#
#                 # Crop the segment from the original image
#                 segment = img.crop((left, upper, right, lower))
#                 segment_number += 1
#                 segment_arr = np.array(segment)[:, :, np.newaxis, np.newaxis]
#                 image_mat = np.concatenate((image_mat, segment_arr), axis=-1) # This concatenation is for the puroposes of the Matlab .predict function
#
#                 arr_temp = np.zeros(segment_number)
#         arr_list.append(arr_temp)
#
# images = np.array(images)
#print(image_mat.shape)
#print(arr_list)


# In[ ]:


# Save the net object that can be loaded (using Matlab) with the deep12_adam_2conv_2sig_2fc_96-96input_10iter_92_1perc.mat file
# After you save the file, you don't need to keep Matlab open

model_data = eng.load('C:/Users/v93777mg/MPhys_project/net.mat', nargout=1) # Directory of the loaded network to be changed
print(model_data)
neural_network_model = model_data['net']

# .predict is Matlab's function, it uses the loaded network and image_mat

output_data = eng.predict(neural_network_model, image_mat_contiguous, nargout=1)
predictions = np.array(output_data._data).flatten()

for i in range(len(predictions)):
    if predictions[i]<0:
        predictions[i] = 0
    elif predictions[i]>1:
        predictions[i] = 1


#print(predictions)


# In[ ]:


# ### Use this code if you want to take the divided images from a folder - in general this module is not needed
# ## Here, the idea is to take the divided images and append them to the arr_list.
# # Then, we assign each segment the corresponding prediction from the last section.

# arr_list = []
# for x in range(start_index, end_index + 1):

#     input_image_directory = "/home/mikolaj/MPhys_project/divided_images_directory_2/" # Directory of the divided pictures to be changes
#     input_image_name = 'mp-' + str(x) + '_segment_'
#     input_image_path = input_image_directory + input_image_name + '0.png'

#     segment_number = 0
#     if not os.path.isfile(input_image_path):
#         print("No such file %s" % input_image_path)
#         continue

#     while os.path.isfile(input_image_path):

#         segment_number += 1
#         input_image_path = input_image_directory + input_image_name + str(segment_number) + '.png'
#         if not os.path.isfile(input_image_path):
#             segment_count.append(segment_number)
#             #print(segment_number)
#             arr_temp = np.zeros(segment_number)
#             arr_list.append(arr_temp)


# In[ ]:


### Here, each segment belonging to arr_list gets assigned its corresponding prediction

current_index = 0
# The loop iterates over the numpy arrays stored in the arr_list
for i in range(len(arr_list)):
    arr_length = len(arr_list[i])
    arr_list[i][:] = predictions[current_index : current_index + arr_length]
    current_index += arr_length
#print(arr_list)


# In[ ]:


for i in range(len(arr_list)):
    arr_list[i] = arr_list[i].reshape((-1, 4))
    arr_list[i] = np.transpose(arr_list[i])
    arr_list[i] = arr_list[i].flatten()

#arr_list
#print(len(predictions))
#print(len(arr_list))
#print(arr_list)


# In[ ]:


## Predictions -> Only one flat segment is required to classify band structure as a flat one
current_index = 0
flat_n = 0
not_flat_n = 0
flat = []
not_flat = []
arr_list3 = arr_list
for i in range(len(arr_list3)):
    arr_length = len(arr_list3[i])
    arr_list3[i][:] = predictions[current_index : current_index + arr_length]
    current_index += arr_length

    flat_band_present = any(element >= flatness_criteria for element in arr_list3[i][:])
    if flat_band_present:
        flat.append(arr_list3[i][:])
    else:
        not_flat.append(arr_list3[i][:])

total_flat = len(flat)
total_not_flat = len(not_flat)
print(f"Arrays with at least one flat segment: {total_flat}")
print(f"Arrays without any flat segments: {total_not_flat}")

fraction = (total_flat / (total_flat + total_not_flat)) * 100
print(f"The fraction of arrays with at least one flat segments: {np.round(fraction)}%")
print("The number of materials analyzed: ", total_flat + total_not_flat)


# In[ ]:


## Predictions -> At least one pair of neighboring flat segments is required to classify the band structure as a flat one
current_index = 0
flat_pairs = []
not_flat_pairs = []

arr_list4 = arr_list
for i in range(len(arr_list4)):
    arr_length = len(arr_list4[i])
    pair_found = False

    # Assuming len(arr_list[i]) > 1 to avoid index out of range for j + 1
    for j in range(len(arr_list4[i]) - 1):
        # Extracting two neighboring elements
        current_element = arr_list4[i][j]
        next_element = arr_list4[i][j + 1]

        # Check if both elements are greater than or equal to 0.5
        if current_element >= flatness_criteria and next_element >= flatness_criteria:
            flat_pairs.append(arr_list4[i][:])
            pair_found = True
            break  # Break out of the loop once a pair is found

    if not pair_found:
        not_flat_pairs.append(arr_list4[i][:])

total_flat = len(flat_pairs)
total_not_flat = len(not_flat_pairs)

print(f"Arrays with at least one pair of neighoring flat segments: {total_flat}")
print(f"Arrays without any pair of neighoring flat segments: {total_not_flat}")

fraction = (total_flat / (total_flat + total_not_flat)) * 100
print(f"The fraction of arrays with at least one pair of neighoring flat segments: {np.round(fraction)}%")


# In[ ]:


## Predictions -> At least one row of all flat segments is required to classify the band structure as a flat one
flat_arrays = []
not_flat_arrays = []

arr_list5 = arr_list
for arr in arr_list5:
    # Reshape the array to have 4 rows and corresponding number of columns
    reshaped_arr = arr.reshape(4, -1)
    
    # Check if at least one row has all elements >= 0.5
    if any(all(element >= flatness_criteria for element in row) for row in reshaped_arr):
        flat_arrays.append(arr)
    else:
        not_flat_arrays.append(arr)

total_flat = len(flat_arrays)
total_not_flat = len(not_flat_arrays)

print(f"Arrays with at least one fully flat row: {total_flat}")
print(f"Arrays without any fully flat row: {total_not_flat}")

# Calculate the fraction
fraction = (total_flat / (total_flat + total_not_flat)) * 100
print(f"The fraction of arrays with at least one fully flat rows: {np.round(fraction)}%")


# In[ ]:


# ### Visualizing all the bands (could be updated)
#
# # # Directory to the folder where the segmented images are
# # input_image_directory = "/home/mikolaj/MPhys_project/divided_images_directory_2/"
#
# ## Display of the full band structures just to check if they are the same after putting up all the segments together - I don't really use it
#
# # for x in range(istart, iend + 1):
# #     input_image_directory = "/home/mikolaj/MPhys_project/divided_images_directory/"
# #     input_image_name = 'mp-' + str(x) + '_segment_'
#
# #     # Create and display the composite image
# #     composite_image = np.zeros((size_row * block_row, size_column * block_column), dtype=np.uint8)
#
# #     for i in range(size_row):
# #         for j in range(size_column):
# #             index = i * size_column + j
# #             file_path = f'{input_image_directory}/{input_image_name}{index}.png'
#
# #             if os.path.isfile(file_path):
# #                 image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
# #                 composite_image[i * block_row: (i + 1) * block_row, j * block_column: (j + 1) * block_column] = image
#
# #     plt.imshow(composite_image, cmap='gray')
# #     plt.title(f'Composite Image for mp-{x}')
# #     plt.show()
#
# images2list = []
# for i in range(len(images)):
#     input_image = images[i]
#     size_row = int(input_image.shape[0] / block_row)
#     size_column = int(input_image.shape[1] / block_column)
#     total_blocks = int(size_row * size_column)
#
#     #divided_image = np.zeros((size_row, size_column, block_row, block_column))
#     divided_image = np.zeros((total_blocks, block_row, block_column))
#     if input_image.shape[0] % block_row == 0 or input_image.shape[1] % block_column == 0:
#         for i in range(size_row):
#             for j in range(size_column):
#                          index = i * size_column + j
#                          divided_image[index] = input_image[(i * block_row):((i + 1) * block_row), (j * block_column):((j + 1) * block_column)]
#                          images2list.append(divided_image[index])
#                          #image = plt.imshow(divided_image[i,j], cmap = 'gray')
#                          # Specify a complete file path including the filename and extension
#                          # Save the image using plt.imsave
#                          #plt.imsave(file_path, divided_image[index], cmap='gray')
#
# images2 = np.array(images2list)
# print(images2.shape)
#
# segments = images2
#
# # Calculate the number of rows and columns for the subplot grid
# num_segments = len(segments)
# num_rows = 4  # You can adjust the number of columns as per your preference
# num_cols = -(-num_segments // num_rows)  # Ceiling division to ensure all segments are covered
# print(num_rows)
# print(num_cols)
#
# fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 4 * num_rows))
# axes = axes.flatten()
# print(num_segments)
#
# # Flatten the 2D array of subplots for easier indexing
# axes = axes.flatten()
#
# ## Display each image segment in a separate subplot - it works only for a single band structure!!
# for i in range(num_segments):
#     extent = [0, 96, 0, 96]  # Extent of the image
#     axes[i].imshow(segments[i], cmap='gray', extent=extent)
#     axes[i].set_title(f'{arr_list2[0][i]:.5f}')
#     print(arr_list2[0][i])
#     axes[i].axis('off')  # Turn off axis labels
#     if arr_list2[0][i] > flatness_criteria:
#         rect = plt.Rectangle((0, 0), 96, 96, edgecolor='red', linewidth=3, fill=False)
#     else:
#         rect = plt.Rectangle((0, 0), 96, 96, edgecolor='black', linewidth=1, fill=False)
#     axes[i].add_patch(rect)
#     #fig.savefig('/home/mikolaj/MPhys_project/report_images/We_mp1.png')
#
# # for i
# #     for i in range(num_segments):
# #         extent = [0, 96, 0, 96]  # Extent of the image
# #         axes[i].imshow(segments[i], cmap='gray', extent=extent)
# #         axes[i].set_title(f'{arr_list2[0][i]:.5f}')
# #         print(arr_list[0][i])
# #         axes[i].axis('off')  # Turn off axis labels
# #         if arr_list2[0][i] > flatness_criteria:
# #             rect = plt.Rectangle((0, 0), 96, 96, edgecolor='red', linewidth=10, fill=False)
# #         else:
# #             rect = plt.Rectangle((0, 0), 96, 96, edgecolor='black', linewidth=1, fill=False)
# #         axes[i].add_patch(rect)



# # In[ ]:


### Histograms

## Compound score -> Calculate the averages of the flatness scores for each row -> end up with 4 flatness scores.
# Then, max of these numbers gives the compound flatness score

arr_list2 = arr_list
compound_flatness_score_all = [] # List of the compound flatness scores for all materials
# Loop iterates over the arrays stored in arr_list
for arr in arr_list2:
    # Reshape the array to have 4 rows and the corresponding number of columns
    reshaped_arr = arr.reshape(4, -1)
    #print(np.shape(reshaped_arr)[1])

    # Check if at least one row has all elements >= flatness_score
    flatness_scores_rows = [] # List of flatness scores for all rows in one band structure
    for row in range(len(reshaped_arr)):
        row_flatness_sum = 0

        for element in reshaped_arr[row]:
            row_flatness_sum += element/np.shape(reshaped_arr)[1] # Average of the flatness scores for each row

        flatness_scores_rows.append(row_flatness_sum)

        compound_flatness_score = max(flatness_scores_rows) # Compound flatness score for a given material

    compound_flatness_score_all.append(compound_flatness_score)

#print(compound_flatness_score_all)

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




# Create a histogram
plt.hist(compound_flatness_score_all, bins=50, color='blue', edgecolor='black')

# Add title and labels
plt.title('Compound Flatness Score Histogram')
plt.xlabel('Compound Flatness Score')
plt.ylabel('Number of materials')

# Show the plot
plt.show()


# In[ ]:


eng.quit() # Quit Matlab engine

