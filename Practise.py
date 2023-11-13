import my_model
import scipy.io
import tensorflow as tf
import numpy as np
import h5py
from tensorflow import keras
import mat73

model = keras.models.load_model('/home/mikolaj/MPhys_project/model_tf.keras')

# Load the .mat file
mat_data1 = scipy.io.loadmat('/home/mikolaj/testset.mat')
mat_data3 = scipy.io.loadmat('/home/mikolaj/testtarget.mat')
mat_data4 = scipy.io.loadmat('/home/mikolaj/traintarget.mat')

with h5py.File('/home/mikolaj/trainset.mat', 'r') as file:
    # Access and work with the data in the file
    dataset = file['trainset']  # Replace with the actual dataset name

    # Convert the dataset to a NumPy array
    numpy_trainset = dataset[()]


# Extract the 4D array from the loaded data
testset = mat_data1['testset']  # Replace 'your_matlab_variable_name' with the actual variable name
testtarget = mat_data3['testtarget']
traintarget = mat_data4['traintarget']


# Convert the MATLAB 4D double array to a NumPy array
numpy_testset = np.array(testset)
numpy_testtarget = np.array(testtarget)
numpy_traintarget = np.array(traintarget)

# Now you can work with 'numpy_4d_array' as a NumPy array

train_examples = np.squeeze(numpy_trainset, axis=1)
train_labels = np.squeeze(numpy_traintarget, axis=0)
test_examples = np.squeeze(numpy_testset, axis=2).transpose(2,0,1)
test_labels = np.squeeze(numpy_testtarget, axis=0)

train_dataset = tf.data.Dataset.from_tensor_slices((train_examples, train_labels))
test_dataset = tf.data.Dataset.from_tensor_slices((test_examples, test_labels))

BATCH_SIZE = 64
SHUFFLE_BUFFER_SIZE = 100

train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
test_dataset = test_dataset.batch(BATCH_SIZE)


model.compile(optimizer = "adam", loss = "binary_crossentropy", metrics=["accuracy"])
model.fit(train_dataset, epochs=20)