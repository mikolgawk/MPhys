from PIL import Image
import numpy as np
import os
import tensorflow as tf
import matplotlib.pyplot as plt

# Load the model you want to use
loaded_model = tf.keras.models.load_model("C:/Users/Czaja/Desktop/MPhys thesis/CNN_model/venv/AI models/test_model_1245.h5")

#directory to the folder where the segmented images are
input_image_directory = "C:/Users/Czaja/Desktop/MPhys thesis/Materials_project_picture_prep/venv/96x96 divided pictures/"


def load_images(start_index, end_index, input_image_directory):
    images = []
    for x in range(start_index, end_index + 1):

        input_image_name = 'band_structure_mp-' + str(x) + '_segment_'
        input_image_path = input_image_directory + input_image_name + '0.png'

        segment_number = 0
        if not os.path.isfile(input_image_path):
            print("No such file %s" % input_image_path)
            continue

        while os.path.isfile(input_image_path):
            image = Image.open(input_image_path)

            # Convert the image to a NumPy array
            image_array = np.array(image)

            # Normalize the pixel values to the range [0, 1] (if needed)
            image_array = image_array / 255.0

            images.append(image_array)

            segment_number += 1

            input_image_path = input_image_directory + input_image_name + str(segment_number) + '.png'

    images = np.array(images)

    return images

def load_images(start_index, end_index, input_image_directory):
    images = []
    for x in range(start_index, end_index + 1):

        input_image_name = 'band_structure_mp-' + str(x) + '_segment_'
        input_image_path = input_image_directory + input_image_name + '0.png'

        segment_number = 0
        if not os.path.isfile(input_image_path):
            print("No such file %s" % input_image_path)
            continue

        while os.path.isfile(input_image_path):
            image = Image.open(input_image_path)

            # Convert the image to a NumPy array
            image_array = np.array(image)

            # Normalize the pixel values to the range [0, 1] (if needed)
            image_array = image_array / 255.0

            images.append(image_array)

            segment_number += 1

            input_image_path = input_image_directory + input_image_name + str(segment_number) + '.png'

    images = np.array(images)

    return images

#Flatness visualisation
def visualise(test_image_id):

    segments = load_images(test_image_id, test_image_id, input_image_directory)
    predictions = loaded_model.predict(segments)

    # Create subplots
    fig, axes = plt.subplots(1, len(segments), figsize=(12, 4))

    # Flatten the 2D array of subplots for easier indexing
    axes = axes.flatten()

    # Display each image segment in a separate subplot
    for i in range(len(segments)):
        extent = [0, 96, 0, 96]  # Extent of the image
        axes[i].imshow(segments[i], cmap='gray', extent=extent)
        axes[i].set_title(f'{predictions[i][1]:.5f}')
        axes[i].axis('off')  # Turn off axis labels
        if predictions[i][1] > 0.5:
            rect = plt.Rectangle((0, 0), 96, 96, edgecolor='red', linewidth=3, fill=False)
        else:
            rect = plt.Rectangle((0, 0), 96, 96, edgecolor='black', linewidth=1, fill=False)
        axes[i].add_patch(rect)

    fig.suptitle('Flatness scores', fontsize=16)
    plt.tight_layout()
    plt.show()

    return 0

def categorize_findings(start_index,end_index, input_image_directory):

    num_of_flat_bands = []
    for index in range(start_index, end_index+1):
        num_of_flat_bands_within_material = 0
        segments = load_images(index, index, input_image_directory)
        try:
            predictions = loaded_model.predict(segments)
        except ValueError:
            continue
        for prediction in predictions:
            if prediction[1] > 0.5:
                num_of_flat_bands_within_material += 1
            else:
                continue
        num_of_flat_bands.append(num_of_flat_bands_within_material)


    num_of_flat_bands = np.array(num_of_flat_bands)
    total_materials = len(num_of_flat_bands)
    total_flat_materials = len(num_of_flat_bands[num_of_flat_bands > 0])
    plt.hist(num_of_flat_bands, bins=np.arange(min(num_of_flat_bands) - 0.5, max(num_of_flat_bands) + 1.5, 1), align='mid', rwidth=0.1, color='skyblue',
             edgecolor='black', label=f"{total_flat_materials / total_materials * 100 :.3f} % of materials were found to have flat bands")
    plt.xlabel('Number of flat bands')
    plt.ylabel('Number of materials')
    plt.title('Flat band frequency')

    # Show the plot
    plt.legend()
    plt.show()
    print(num_of_flat_bands)

    return 0

#prototype function
def histogram(predictions):
    return 0
#Accuracy test

def test_accuracy(test_images, known_outcomes):
    #Test over a dataset
    test_loss, test_accuracy = loaded_model.evaluate(test_images, known_outcomes) #input many pictures with labels to see how accurate it is

    print("Test loss: ")
    print(test_loss)
    print("Test accuracy: ")
    print(test_accuracy)
    return 0


#Single prediction
def single_prediction(image_name, input_image_directory):
    prediction_image_path = input_image_directory + image_name

    predictions = loaded_model.predict(np.array([np.array(Image.open(prediction_image_path))])) #You get response in this form: [not flat probability, flat probability)
    print(predictions)
    return predictions



## Run what you want (examples given below)

#test_accuracy(numpy_array_of_images, numpy_array_of_assigned_outcomes)
"""
test_accuracy(load_images(447,499, input_image_directory), np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                               1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                               1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0,
                               1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1,
                               1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0,
                               1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0,
                               1, 0, 0, 0, 0]))
"""
#single_prediction("band_structure_mp-454_segment_4.png",input_image_directory)

#visualise(454)
#categorize_findings(447,499, input_image_directory)