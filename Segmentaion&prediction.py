### Import libraries
import numpy as np
import os
import cv2
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from PIL import Image

### Define constants

block_column = 96 # Number of pixels in a column
block_row = 96 # Number of pixels in a row
istart = 1 # start index in material ids
iend = 10 # end index in material ids (inclusive)
flatness_criteria = 0.5 # flatness criterion based on the histograms

### Input and segment the images

for x in range(istart, iend + 1):
    try:
        directory = f'/home/mikolaj/MPhys_project/2dmatpedia_images/' # Change the directory
        band_name = "2dm-" + str(x) + ".png" # If it is materials project id, then change "2dm-" to "mp-"
        file = directory + band_name
        lst = cv2.imread(file) # Use cv2 library to read the images
        image_g = cv2.cvtColor(lst, cv2.COLOR_BGR2GRAY)
        input_image = np.array(image_g)


        size_row = int(input_image.shape[0] / block_row)
        size_column = int(input_image.shape[1] / block_column)
        total_blocks = int(size_row * size_column)
        
        divided_image = np.zeros((total_blocks, block_row, block_column)) # divided_image stores the divided segments where total_blocks is the number of them
        
        if input_image.shape[0] % block_row == 0 or input_image.shape[1] % block_column == 0:
            for i in range(size_row):
                for j in range(size_column):
                     index = i * size_column + j
                     divided_image[index] = input_image[(i * block_row):((i + 1) * block_row), (j * block_column):((j + 1) * block_column)]
                     # Specify a complete file path including the filename and extension
                     file_path = f'{"/home/mikolaj/MPhys_project/divided_images_2dmatpedia/"}/' + f'{"2dm-"}' + f'{str(x)}' + f'_segment_{index}.png'
                     # Save the image using plt.imsave
                     plt.imsave(file_path, divided_image[index], cmap='gray')
    except:
            pass


# Create and display the composite image
composite_image = np.zeros((size_row * block_row, size_column * block_column), dtype=np.uint8)

for i in range(size_row):
    for j in range(size_column):
        index = i * size_column + j
        composite_image[i * block_row: (i + 1) * block_row, j * block_column: (j + 1) * block_column] = divided_image[index]

plt.imshow(composite_image, cmap='gray')
plt.title('Composite Image of All Blocks')
plt.show()

### Store the segments in a numpy array

images = []

for x in range(istart, iend + 1):

    input_image_directory = "/home/mikolaj/MPhys_project/divided_images_2dmatpedia/"
    input_image_name = '2dm-' + str(x) + '_segment_' # If it is materials project id, then change "2dm-" to "mp-"
    input_image_path = input_image_directory + input_image_name + '0.png'

    segment_number = 0
    #row_number = 0
    #column_number = 0
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


images = np.array(images) #train image data in the format [image_1_array, image_2_array, ....

### Predict and save the segments

model = keras.models.load_model('/home/mikolaj/MPhys_project/model_segmentation.keras') # Load the CNN
test_set = images
predictions = model.predict(test_set)
#print(predictions) # the first number is not flat probability, the second one is flat probability

segment_count = []
arr_list = []

for x in range(istart, iend + 1):

    input_image_directory = "/home/mikolaj/MPhys_project/divided_images_2dmatpedia/"
    input_image_name = '2dm-' + str(x) + '_segment_' # If it is materials project id, then change "2dm-" to "mp-"
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
            arr_list.append(arr_temp)
            

probabilities = predictions[:,1]

current_index = 0
for i in range(len(arr_list)):
    arr_length = len(arr_list[i])
    arr_list[i][:] = probabilities[current_index : current_index + arr_length]
    current_index += arr_length

print(arr_list)


### Predictions -> only one flat segment
current_index = 0
flat_n = 0
not_flat_n = 0
flat = []
not_flat = []
for i in range(len(arr_list)):
    arr_length = len(arr_list[i])
    arr_list[i][:] = probabilities[current_index : current_index + arr_length]
    current_index += arr_length

    flat_band_present = any(element >= 0.5 for element in arr_list[i][:])
    if flat_band_present:
        flat.append(arr_list[i][:])
    else:
        not_flat.append(arr_list[i][:])

total_flat = len(flat)
total_not_flat = len(not_flat)

fraction = (total_flat / (total_flat + total_not_flat))*100
print("The ratio is: ", np.round(fraction), "%")

# Visualize the images with predicted flat bands
current_index = 0
for x in range(istart, iend + 1):
    input_image_name = '2dm-' + str(x) + '_segment_'
    input_image_path = input_image_directory + input_image_name + '0.png'

    segment_number = 0
    if not os.path.isfile(input_image_path):
        print("No such file %s" % input_image_path)
        continue

    while os.path.isfile(input_image_path):
        segment_number += 1
        input_image_path = input_image_directory + input_image_name + str(segment_number) + '.png'

        if not os.path.isfile(input_image_path):
            # Visualization
            fig, ax = plt.subplots()
            ax.imshow(images[current_index], cmap='gray')
            ax.set_title(f'Segment {segment_number} - Probability: {probabilities[current_index]:.2f}')

            # Highlight frame in red for segments with probability >= 0.5
            if probabilities[current_index] >= 0.5:
                for spine in ax.spines.values():
                    spine.set_edgecolor('red')
                    spine.set_linewidth(4)

            plt.show()

        current_index += 1

### Preidctions -> two neighbouring flat segments
# current_index = 0
# flat_pairs = []
# not_flat_pairs = []

# for i in range(len(arr_list)):
#     arr_length = len(arr_list[i])

#     # Assuming len(arr_list[i]) > 1 to avoid index out of range for j + 1
#     for j in range(len(arr_list[i]) - 1):
#         # Extracting two neighboring elements
#         current_element = arr_list[i][j]
#         next_element = arr_list[i][j + 1]

#         # Check if both elements are greater than or equal to 0.5
#         if current_element >= 0.5 and next_element >= 0.5:
#             flat_pairs.append((current_element, next_element))
#         else:
#             not_flat_pairs.append((current_element, next_element))

# total_flat = len(flat_pairs)
# total_not_flat = len(not_flat_pairs)

# fraction = (total_flat / (total_flat + total_not_flat))*100
# print("The ratio is: ", np.round(fraction), "%")

### Predictions -> the entire horizontal band being flat
# current_index = 0
# flat_n = 0
# not_flat_n = 0
# flat = []
# not_flat = []
# for i in range(len(arr_list)):
#     arr_length = len(arr_list[i])
#     arr_list[i][:] = probabilities[current_index : current_index + arr_length]
#     current_index += arr_length

#     flat_band_present = all(element >= 0.5 for element in arr_list[i][:])
#     if flat_band_present:
#         flat.append(arr_list[i][:])
#     else:
#         not_flat.append(arr_list[i][:])

# total_flat = len(flat)
# total_not_flat = len(not_flat)

# fraction = (total_flat / (total_flat + total_not_flat))*100
# print("The ratio is: ", np.round(fraction), "%")

