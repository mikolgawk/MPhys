Anupam's CNN layer details:
Image input required 96x96x1 (Worth considering converting pictures to greyscale to have 96x96, as we did in the new TensorFlow AI)
1. Convolutional Layer, filter size = 10x10, stride = [1,1], number of filters = 30.
2. Sigmoid
3. Convolutional Layer, filter size = 3x3, stride = [1,1], number of filters = 12.
4. Sigmoid
5. Fully Connected (Dense) Layer (output should be 80 neurons)
6. Sigmoid
7. Fully Connected (Dense) Layer (output should be 1 neuron, giving the not normalized flatness score)
8. Regression, loss function: mean squared error
