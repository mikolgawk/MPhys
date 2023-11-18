#!/usr/bin/env python
# coding: utf-8

# In[236]:


import numpy as np
import os
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow import keras


# In[237]:


block_column = 96
block_row = 96
istart = 6000
iend = 6000
flatness_criteria = 0.5


# In[238]:

## Input the images

for x in range(istart, iend + 1):
    try:
        directory = f'/home/mikolaj/MPhys_project/band_structure_pictures/' ## Change directory
        band_name = "mp-" + str(x) + ".png" ## I'm using 'mp-' instead of 'band_structure_mp-'
        file = directory + band_name
        lst = cv2.imread(file)
        image_g = cv2.cvtColor(lst, cv2.COLOR_BGR2GRAY)
    
        input_image = np.array(image_g)

        ## Now I want to do this for all images

        size_row = int(input_image.shape[0] / block_row)
        size_column = int(input_image.shape[1] / block_column)
        total_blocks = int(size_row * size_column)
        
        divided_image = np.zeros((total_blocks, block_row, block_column))
        if input_image.shape[0] % block_row == 0 or input_image.shape[1] % block_column == 0:
            for i in range(size_row):
                for j in range(size_column):
                     index = i * size_column + j
                     divided_image[index] = input_image[(i * block_row):((i + 1) * block_row), (j * block_column):((j + 1) * block_column)]
                     # Specify a complete file path including the filename and extension
                     file_path = f'{"/home/mikolaj/MPhys_project/divided_images_directory/"}/' + f'{"mp-"}' + f'{str(x)}' + f'_segment_{index}.png' ## Change directory
                     # Save the image using plt.imsave
                     plt.imsave(file_path, divided_image[index], cmap='gray')
    except:
            pass

# Create and display the composite image (not neccessary)
composite_image = np.zeros((size_row * block_row, size_column * block_column), dtype=np.uint8)

for i in range(size_row):
    for j in range(size_column):
        index = i * size_column + j
        composite_image[i * block_row: (i + 1) * block_row, j * block_column: (j + 1) * block_column] = divided_image[index]

plt.imshow(composite_image, cmap='gray')
plt.title('Composite Image of All Blocks')
plt.show()


# In[239]:


images = []

for x in range(istart, iend + 1):

    input_image_directory = "/home/mikolaj/MPhys_project/divided_images_directory/" ## Change directory
    input_image_name = 'mp-' + str(x) + '_segment_'
    input_image_path = input_image_directory + input_image_name + '0.png'

    segment_number = 0
    if not os.path.isfile(input_image_path):
        print("No such file %s" % input_image_path)
        continue

    while os.path.isfile(input_image_path):
        image = Image.open(input_image_path)
        image = image.convert('L')
        # Convert the image to a NumPy array
        image_array = np.array(image)

        # Normalize the pixel values to the range [0, 1] (if needed)
        image_array = image_array / 255.0


        segment_number += 1

        images.append(image_array)
        input_image_path = input_image_directory + input_image_name + str(segment_number) + '.png'


images = np.array(images)#train image data in the format [image_1_array, image_2_array, ....


# In[240]:


model = keras.models.load_model('/home/mikolaj/MPhys_project/test_model_1245.h5') ## Change directory
test_set = images
predictions = model.predict(test_set)

## Algorithm
segment_count = []

probs_list = []
for x in range(istart, iend + 1):

    input_image_name = 'mp-' + str(x) + '_segment_'
    input_image_path = input_image_directory + input_image_name + '0.png'

    segment_number = 0

    if not os.path.isfile(input_image_path):
        print("No such file %s" % input_image_path)
        continue

    while os.path.isfile(input_image_path):

        segment_number += 1

        input_image_path = input_image_directory + input_image_name + str(segment_number) + '.png'

        if not os.path.isfile(input_image_path):
            segment_count.append(segment_number)
            arr_temp = np.zeros(segment_number)
            probs_list.append(arr_temp)
            

probabilities = predictions[:,1]

current_index = 0
for i in range(len(probs_list)):
    arr_length = len(probs_list[i])
    probs_list[i][:] = probabilities[current_index : current_index + arr_length]
    current_index += arr_length

#print(probs_list)


# In[241]:


## Predictions -> only one flat segment
current_index = 0
flat = []
not_flat = []

for i in range(len(probs_list)):
    arr_length = len(probs_list[i])
    probs_list[i][:] = probabilities[current_index : current_index + arr_length]
    current_index += arr_length

    # Check if any segment in a given band is flat
    flat_band_present = any(element >= flatness_criteria for element in probs_list[i][:])
    if flat_band_present:
        flat.append(probs_list[i][:])
    else:
        not_flat.append(probs_list[i][:])

total_flat = len(flat)
total_not_flat = len(not_flat)

fraction = (total_flat / (total_flat + total_not_flat)) * 100
print("The fraction of materials having at least one flat segment is: ", np.round(fraction, 2), "%")
print("The number of analyzed materials is: ", total_flat + total_not_flat)



# In[242]:


## Pairs of flat segments
current_index = 0
flat_pairs = []
not_flat_pairs = []

for i in range(len(probs_list)):
    arr_length = len(probs_list[i])
    pair_found = False

    # Assuming len(arr_list[i]) > 1 to avoid index out of range for j + 1
    for j in range(len(probs_list[i]) - 1):
        # Extracting two neighboring elements
        current_element = probs_list[i][j]
        next_element = probs_list[i][j + 1]

        # Check if both elements are greater than or equal to flatness_criteria
        if current_element >= flatness_criteria and next_element >= flatness_criteria:
            flat_pairs.append(probs_list[i][:])
            pair_found = True
            break  # Break out of the loop once a pair is found

    if not pair_found:
        not_flat_pairs.append(probs_list[i][:])

total_flat = len(flat_pairs)
total_not_flat = len(not_flat_pairs)

fraction = (total_flat / (total_flat + total_not_flat)) * 100
print("The fraction of materials having at least one pair of flat segments is: ", np.round(fraction, 2), "%")


# In[243]:


### Predictions -> the entire horizontal band being flat
flat_arrays = []
not_flat_arrays = []

for arr in probs_list:
    # Reshape the array to have 4 rows and corresponding number of columns
    reshaped_arr = arr.reshape(4, -1)
    
    # Check if at least one row has all elements >= flatness_criteria
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
print(f"The fraction of arrays with at least one fully flat row: {np.round(fraction)}%")


# In[255]:


# ### Visualizing all the bands (doesn't work well)
# # Load the model you want to use
# loaded_model = tf.keras.models.load_model("/home/mikolaj/MPhys_project/test_model_1245.h5")

# # Directory to the folder where the segmented images are
# input_image_directory = "/home/mikolaj/MPhys_project/divided_images_directory/"

# # Flatness visualization
# segments = images
# predictions = model.predict(images)


# for x in range(istart, iend + 1):
#     input_image_directory = "/home/mikolaj/MPhys_project/divided_images_directory/"
#     input_image_name = 'mp-' + str(x) + '_segment_'
    
#     # Create and display the composite image
#     composite_image = np.zeros((size_row * block_row, size_column * block_column), dtype=np.uint8)

#     for i in range(size_row):
#         for j in range(size_column):
#             index = i * size_column + j
#             file_path = f'{input_image_directory}/{input_image_name}{index}.png'
            
#             if os.path.isfile(file_path):
#                 image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
#                 composite_image[i * block_row: (i + 1) * block_row, j * block_column: (j + 1) * block_column] = image

#     plt.imshow(composite_image, cmap='gray')
#     plt.title(f'Composite Image for mp-{x}')
#     plt.show()
    
# # Calculate the number of rows and columns for the subplot grid
# num_segments = len(segments)
# num_rows = 4  # You can adjust the number of columns as per your preference
# num_cols = -(-num_segments // num_rows)  # Ceiling division to ensure all segments are covered
# print(num_rows)

# fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 4 * num_rows))
# axes = axes.flatten()
# print(num_segments)

# # # Flatten the 2D array of subplots for easier indexing
# axes = axes.flatten()

# # Display each image segment in a separate subplot
# for i in range(num_segments):
#     extent = [0, 96, 0, 96]  # Extent of the image
#     axes[i].imshow(segments[i], cmap='gray', extent=extent)
#     axes[i].set_title(f'{predictions[i][1]:.5f}')
#     axes[i].axis('off')  # Turn off axis labels
#     if predictions[i][1] > 0.5:
#         rect = plt.Rectangle((0, 0), 96, 96, edgecolor='red', linewidth=3, fill=False)
#     else:
#         rect = plt.Rectangle((0, 0), 96, 96, edgecolor='black', linewidth=1, fill=False)
#     axes[i].add_patch(rect)



# In[254]:


### Histograms (don't know for sure how the compound flatness score is defined)

num_of_flat_bands = []
compound_score = []
compound_score_sum = []
segments = images

for arr in probs_list:
    sum = 0
    num_of_flat_bands_within_material = 0
    for i in range(len(arr)):
        sum += arr[i]
        if arr[i] >= flatness_criteria:
            num_of_flat_bands_within_material += 1
    compound_score_sum.append(sum)
    num_of_flat_bands.append(num_of_flat_bands_within_material)
    

num_of_flat_bands = np.array(num_of_flat_bands)
compound_score = compound_score_sum / num_of_flat_bands

print("The compound flatness score is: ", np.round(compound_score, 2))

# Create a histogram
plt.hist(compound_score, bins=30, color='blue', edgecolor='black')

# Add title and labels
plt.title('Compound Flatness Score Histogram')
plt.xlabel('Compound Flatness Score')
plt.ylabel('Number of materials')

# Show the plot
plt.show()
