# Backpropagation

iteratively adjust the weights to empirically minimize error (some measure of vector differences)

the key idea is that backpropagation creates useful new features

### simple form of learning procedure

one input layer, any number of intermediate layers, one output layer
- strictly feedforward
- neurons in layers don't necessarily need to connect to the next one

hidden units in the same layer have states set in parallel-- different layers set sequentially

total input to a neuron is a linear function of outputs from previous layer.

a bias on a unit is optional and is applied by introducing an extra input of 1
- the corresponding weight is the bias

exact linear functions and sigmoid is kind of arbitrary but linear inputs and nonlinear outputs simplifies the process.

total error can be computed by measuring the loss for every sample.

the goal is to minimize total error by gd...
- NEED GRADIENTS OF THE LOSS
--> must compute partial derivative of E with respect to every weight in the network

partial derivatives found by chain rule & summing up each squared norm of error vector

### backward pass

compute the partial derivative of the error with respect to each output unit y

combine terms using partial derivative rules to get the loss with respect to each weight for every training sample.
--> continue this pattern repeatedly for successively earlier layers

