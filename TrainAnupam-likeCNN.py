import numpy as np
import h5py
from tensorflow import keras

#number of epochs (how many times you repeat the training over the data_set)
num_of_epochs = 20

#batch size (how many pictures you take to do one coefficient reevaluation)
batch_size = 3

def set_up_the_CNN():
    #I set it up Anupam's way (I think Anupam did 1 number of outcomes)
    number_of_possible_outcomes = 2  # Flat or not
    # The CNN model
    model = keras.Sequential([
        keras.layers.Conv2D(30, (10, 10), activation='relu', input_shape=(96, 96, 1)),
        #keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(12, (3, 3), activation='relu'),
        keras.layers.Flatten(),
        keras.layers.Dense(80, activation='relu'),
        keras.layers.Dense(number_of_possible_outcomes, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',  # Use 'sparse_categorical_crossentropy' for integer labels
                  metrics=['accuracy'])

    return model

def train_CNN(num_of_epochs, batch_size):
    pre_assigned_outcomes = import_training_labels('traintarget.txt')
    images = load_anupams_pictures('trainset')

    model = set_up_the_CNN()
    model.fit(images, pre_assigned_outcomes, epochs=num_of_epochs, batch_size=batch_size)
    model.save("C:/Users/Czaja/Desktop/MPhys thesis/CNN_model/venv/AI models/Anupams_test_model.h5")
    return 0

def import_training_labels(filename):
    #Converts training labels from .txt to numpy array

    #Load the target data from the text file
    traintarget = np.loadtxt(filename, delimiter=',')

    #Ensure traintarget is in the desired format (1D NumPy array)
    traintarget = traintarget.squeeze()

    print(traintarget) #Numpy array of zeros and ones
    return traintarget

def load_anupams_pictures(filename):
    # Load the MATLAB file using h5py
    mat_file = h5py.File(filename +'.mat', 'r')

    # Extract the 4-D double array
    mat_array = mat_file[filename][:]  # Replace 'your_variable_name' with the actual variable name in the MATLAB file

    # Convert the MATLAB array to a NumPy array
    numpy_array = np.array(mat_array)

    numpy_array = np.squeeze(numpy_array, axis=1)

    # Verify the shape of the NumPy array
    print(numpy_array.shape)

    # Close the file
    mat_file.close()
    return numpy_array


train_CNN(num_of_epochs, batch_size)
