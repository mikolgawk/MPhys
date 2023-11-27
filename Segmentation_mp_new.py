#!/usr/bin/env python
# coding: utf-8

# In[293]:


import numpy as np
import os
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import matlab.engine
#import matlab
from PIL import Image


# In[294]:


block_column = 96
block_row = 96
start_index = 1
end_index = 40
flatness_criteria = 0.5


# In[295]:


eng = matlab.engine.start_matlab() # Start Matlab engine


# In[296]:


## Segmentation module
# First, initialize an empty array where the third dim is the colour number (basically representing the .convert('L') function),
# and the last dim is the total number of segments that is then appended in the for loop

## This code does not save the divided images to avoid storing thousands of them in one folder

image_mat = np.empty((96, 96, 1, 0), dtype=np.uint8)


arr_list = []
for x in range(start_index, end_index + 1):
        directory = f'/home/mikolaj/MPhys_project/band_structure_pictures/' # Directory of the full images to be changed
        band_name = "mp-" + str(x) + ".png"
        input_image_path = directory + band_name

        if not os.path.isfile(input_image_path):
            print("No such file %s" % input_image_path)
            continue
        
    
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
        for i in range(num_horizontal_segments):
            for j in range(num_vertical_segments):
                # Calculate the cropping region for the current segment
                left = i * target_size[0]
                upper = j * target_size[1]
                right = (i + 1) * target_size[0]
                lower = (j + 1) * target_size[1]
    
                # Crop the segment from the original image
                segment = img.crop((left, upper, right, lower))
                segment_arr = np.array(segment)[:, :, np.newaxis, np.newaxis]               
                image_mat = np.concatenate((image_mat, segment_arr), axis=-1) # This concatenation is for the puroposes of the Matlab .predict function        

                segment_number += 1
                arr_temp = np.zeros(segment_number)
        arr_list.append(arr_temp)
        
#print(arr_list)


# In[297]:


# Save the net object that can be loaded (using Matlab) with the deep12_adam_2conv_2sig_2fc_96-96input_10iter_92_1perc.mat file
# After you save the file, you don't need to keep Matlab open

model_data = eng.load('/home/mikolaj/net.mat', nargout=1) # Directory of the loaded network to be changed
print(model_data)
neural_network_model = model_data['net']

# .predict is Matlab's function, it uses the loaded network and image_mat

output_data = eng.predict(neural_network_model, image_mat, nargout=1)
predictions = np.array(output_data._data).flatten()
#print(predictions)


# In[298]:


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


# In[299]:


### Here, each segment belonging to arr_list gets assigned its corresponding prediction

current_index = 0
# The loop iterates over the numpy arrays stored in the arr_list
for i in range(len(arr_list)):
    arr_length = len(arr_list[i])
    arr_list[i][:] = predictions[current_index : current_index + arr_length]
    current_index += arr_length

#print(len(predictions))
#print(len(arr_list))
#print(arr_list)


# In[300]:


## Predictions -> Only one flat segment is required to classify band structure as a flat one
current_index = 0
flat_n = 0
not_flat_n = 0
flat = []
not_flat = []
for i in range(len(arr_list)):
    arr_length = len(arr_list[i])
    arr_list[i][:] = predictions[current_index : current_index + arr_length]
    current_index += arr_length

    flat_band_present = any(element >= 0.5 for element in arr_list[i][:])
    if flat_band_present:
        flat.append(arr_list[i][:])
    else:
        not_flat.append(arr_list[i][:])

total_flat = len(flat)
total_not_flat = len(not_flat)
print(f"Arrays with at least one flat segment: {total_flat}")
print(f"Arrays without any flat segments: {total_not_flat}")

fraction = (total_flat / (total_flat + total_not_flat)) * 100
print(f"The fraction of arrays with at least one flat segments: {np.round(fraction)}%")
print("The number of materials analyzed: ", total_flat + total_not_flat)


# In[301]:


## Predictions -> At least one pair of neighboring flat segments is required to classify the band structure as a flat one
current_index = 0
flat_pairs = []
not_flat_pairs = []

for i in range(len(arr_list)):
    arr_length = len(arr_list[i])
    pair_found = False

    # Assuming len(arr_list[i]) > 1 to avoid index out of range for j + 1
    for j in range(len(arr_list[i]) - 1):
        # Extracting two neighboring elements
        current_element = arr_list[i][j]
        next_element = arr_list[i][j + 1]

        # Check if both elements are greater than or equal to 0.5
        if current_element >= 0.5 and next_element >= 0.5:
            flat_pairs.append(arr_list[i][:])
            pair_found = True
            break  # Break out of the loop once a pair is found

    if not pair_found:
        not_flat_pairs.append(arr_list[i][:])

total_flat = len(flat_pairs)
total_not_flat = len(not_flat_pairs)

print(f"Arrays with at least one pair of neighoring flat segments: {total_flat}")
print(f"Arrays without any pair of neighoring flat segments: {total_not_flat}")

fraction = (total_flat / (total_flat + total_not_flat)) * 100
print(f"The fraction of arrays with at least one pair of neighoring flat segments: {np.round(fraction)}%")


# In[302]:


## Predictions -> At least one row of all flat segments is required to classify the band structure as a flat one
flat_arrays = []
not_flat_arrays = []

for arr in arr_list:
    # Reshape the array to have 4 rows and corresponding number of columns
    reshaped_arr = arr.reshape(4, -1)
    
    # Check if at least one row has all elements >= 0.5
    if any(all(element >= 0.5 for element in row) for row in reshaped_arr):
        flat_arrays.append(arr)
    else:
        not_flat_arrays.append(arr)

total_flat = len(flat_arrays)
total_not_flat = len(not_flat_arrays)

print(f"Arrays with at least one fully flat row: {total_flat}")
print(f"Arrays without any fully flat row: {total_not_flat}")

# Calculate the fraction
fraction = (total_flat / (total_flat + total_not_flat)) * 100
print(f"The fraction of arrays with at least one fully flat row: {np.round(fraction)}%")


# In[303]:


# ### Visualizing all the bands (could be updated)

# # Directory to the folder where the segmented images are
# input_image_directory = "/home/mikolaj/MPhys_project/divided_images_directory_2/"

# ## Display of the full band structures just to check if they are the same after putting up all the segments together - I don't really use it

# # for x in range(istart, iend + 1):
# #     input_image_directory = "/home/mikolaj/MPhys_project/divided_images_directory/"
# #     input_image_name = 'mp-' + str(x) + '_segment_'
    
# #     # Create and display the composite image
# #     composite_image = np.zeros((size_row * block_row, size_column * block_column), dtype=np.uint8)

# #     for i in range(size_row):
# #         for j in range(size_column):
# #             index = i * size_column + j
# #             file_path = f'{input_image_directory}/{input_image_name}{index}.png'
            
# #             if os.path.isfile(file_path):
# #                 image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
# #                 composite_image[i * block_row: (i + 1) * block_row, j * block_column: (j + 1) * block_column] = image

# #     plt.imshow(composite_image, cmap='gray')
# #     plt.title(f'Composite Image for mp-{x}')
# #     plt.show()


# # Calculate the number of rows and columns for the subplot grid
# num_segments = len(image_mat)
# num_rows = 4  # You can adjust the number of columns as per your preference
# num_cols = -(-num_segments // num_rows)  # Ceiling division to ensure all segments are covered
# print(num_rows)

# fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 4 * num_rows))
# axes = axes.flatten()
# print(num_segments)

# # Flatten the 2D array of subplots for easier indexing
# axes = axes.flatten()

# ## Display each image segment in a separate subplot - it works only for a single band structure!!
# for i in range(num_segments):
#     extent = [0, 96, 0, 96]  # Extent of the image
#     axes[i].imshow(image_mat[i], cmap='gray', extent=extent)
#     axes[i].set_title(f'{predictions[i]:.5f}')
#     axes[i].axis('off')  # Turn off axis labels
#     if predictions[i] > 0.5:
#         rect = plt.Rectangle((0, 0), 96, 96, edgecolor='red', linewidth=3, fill=False)
#     else:
#         rect = plt.Rectangle((0, 0), 96, 96, edgecolor='black', linewidth=1, fill=False)
#     axes[i].add_patch(rect)



# In[304]:


### Histograms

## Compound score -> Calculate the averages of the flatness scores for each row -> end up with 4 flatness scores.
# Then, max of these numbers gives the compound flatness score

compound_flatness_score_all = [] # List of the compound flatness scores for all materials
# Loop iterates over the arrays stored in arr_list
for arr in arr_list:
    # Reshape the array to have 4 rows and the corresponding number of columns
    reshaped_arr = arr.reshape(4, -1)
    
    # Check if at least one row has all elements >= flatness_score
    flatness_scores_rows = [] # List of flatness scores for all rows in one band structure
    for row in range(len(reshaped_arr)):
        row_flatness_sum = 0
        
        for element in reshaped_arr[row]:
            row_flatness_sum += element/num_cols # Average of the flatness scores for each row
            
        flatness_scores_rows.append(row_flatness_sum) 
        
        compound_flatness_score = max(flatness_scores_rows) # Compound flatness score for a given material
        
    compound_flatness_score_all.append(compound_flatness_score)

#print(compound_flatness_score_all)

# Create a histogram
plt.hist(compound_flatness_score_all, bins=50, color='blue', edgecolor='black')

# Add title and labels
plt.title('Compound Flatness Score Histogram')
plt.xlabel('Compound Flatness Score')
plt.ylabel('Number of materials')

# Show the plot
plt.show()


# In[305]:


eng.quit() # Quit Matlab engine

