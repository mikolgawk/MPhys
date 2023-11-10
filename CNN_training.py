from PIL import Image
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras

#number of epochs (how many times you repeat the training over the data_set)
num_of_epochs = 20

#batch size (how many pictures you take to do one coefficient reevaluation)
batch_size = 3

# Open the images
start_index = 1 #inclusive
end_index = 4 #inclusive
images = []
for x in range(start_index, end_index + 1):

    input_image_directory = "C:/Users/Czaja/Desktop/MPhys thesis/Materials_project_picture_prep/venv/96x96 divided pictures/"
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


images = np.array(images) #train image data in the format [image_1_array, image_2_array, ....]
pre_assigned_outcomes = np.array([0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0])

number_of_possible_outcomes = 2 #Flat or not flat

# The CNN model
model = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(96, 96, 1)),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation='sigmoid'),
    keras.layers.Dense(number_of_possible_outcomes, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',  # Use 'sparse_categorical_crossentropy' for integer labels
              metrics=['accuracy'])

model.fit(images, pre_assigned_outcomes, epochs=num_of_epochs, batch_size=batch_size)


#Testing the model accuracy on many pictures
#test_loss, test_accuracy = model.evaluate(test_images, test_labels) #input many pictures with labels to see how accurate it is

#Single prediction
predictions = model.predict(np.array([image_array])) # You get response in this form: [not flat probability, flat probability)

print(predictions)

