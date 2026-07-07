# ResNet

**problem**: deeper neural nets exhibit worse TRAINING error than less deep ones

- later justified to be shattered gradients

deep cnns were super effective for image classification, but to what extent?

vanishing/exploding gradients were largely solved by initialization and batch
norm

the degradation problem is when training accuracy decreases with network depth

### the solution to degradation

a deep residual learning framework

- WHY? it's easier to optimize residual mappings than unreferenced functions
    - (e.g., easier to represent identity mappings instead of learning it)

implement the residual mapping by adding an identity mapping to the learned
function

- adds NO extra params and computationally equivalent

### deep residual learning

instead of approximating an underlying mapping of input to output, approximate a
residual function (wrt the input)

- by reformulation, the optimal mapping is our approximated function plus the
  input (additive)

if the input dimension is larger than the function dimension, project the input
x onto the function output space (by using a learned matrix W)
