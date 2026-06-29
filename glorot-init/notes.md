# Glorot Init

**problem**: regular gd is not working well with deep networks

found that sigmoid is not suitable to use because of mean value of activations
- causes saturation

shows how activations and gradients change over layers (in magnitude and variance)

**solution**: a new way of initializing parameters 
- faster convergence

### activation functions & saturation

#### sigmoid
excessively saturated activation functions don't propagate gradients well (gradients vanish)

when sigmoid saturates in later layers, hard to propagate gradients back to earlier layers

#### tanh
tanh networks don't suffer from the top layer saturation problem in sigmoid networks because of zero mean weight init

but saturation occurs sequentially over time
- early layers saturate more strongly

### gradients & propagation

weights should be initialized randomly from a distribution where variance of weights & variance of gradients wrt weights is all the same.

normalizing the initialized weights fixes this problem of gradients vanishing during backward pass

---
jacobian singular values near 1 is our proxy for good gradient flow.

