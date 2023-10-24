## This file loads train set and test set from matlab and converts them to numpy arrays.
## It then trains the data

## Import libraries
import numpy as np
import tensorflow as tf
import scipy.io # used to convert matlab arrays to numpy ones
import h5py # used to convert matlab arrays to numpy ones

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

## Post-processing of the arrays
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

# model = tf.keras.Sequential([
#     tf.keras.layers.Conv2D(30, (10, 10), input_shape=(96,96,1)),
#     tf.keras.layers.Conv2D(12, (3, 3), input_shape=(96,96,1)),
#     tf.keras.layers.Dense(80),
#     tf.keras.layers.Dense(1)
# ])

# model = tf.keras.models.Sequential([tf.keras.layers.Conv2D(16, (3, 3), activation = 'relu', input_shape = (96, 96, 1)),
#                                    tf.keras.layers.MaxPool2D(2, 2),
#                                    tf.keras.layers.Conv2D(32, (3, 3), activation = 'relu', input_shape = (96, 96, 1)),
#                                    tf.keras.layers.MaxPool2D(2, 2),
#                                    tf.keras.layers.Conv2D(64, (3, 3), activation = 'relu', input_shape = (96, 96, 1)),
#                                    tf.keras.layers.MaxPool2D(2, 2), tf.keras.layers.Flatten(), tf.keras.layers.Dense(512,activation='relu'), tf.keras.layers.Dense(1,activation='sigmoid')])


model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(96, 96)),
    tf.keras.layers.Dense(128, activation='sigmoid'),
    tf.keras.layers.Dense(10)
])

# model.compile(optimizer="adam",
#               loss="mean_squared_error",
#               metrics='accuracy'
#               )

model.compile(optimizer=tf.keras.optimizers.RMSprop(),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['sparse_categorical_accuracy'])

model.fit(train_dataset, epochs=5)

model.evaluate(test_dataset)
