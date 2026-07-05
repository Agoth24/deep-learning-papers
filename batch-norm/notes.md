# Batch Normalization

inputs to each layer of deep networks follow different distributions at each
training iteraton due to changing parameters in the backward pass

- slows down training for deep networks
- INTERNAL COVARIATE SHIFT

batch norm normalizes layer inputs for each mini batch

- allows for higher learning rates, looser initialization, performs
  regularization

when the inputs to a layer change, the parameters at that layer have to be
learned even more than necessary due to changing distribution of the input x

- the input is a combo of all previous layer input and parameterized functions

### mechanisms of reducing ICS

internal covariate shift is the change in activations' distribution due to
changes in parameters.

LeCun (1998) found that zero mean & unit variance inputs (whitening) allow for
faster convergence

the goal is to have all layer inputs come from the same distribution for any
parameter values

### normalization using mini-batch stats

normalize the input vectors componentwise using the mean and variance from the
entire training set ideally (mini-batch in practice).

to allow for the identity transform, bring in scale and shift params for the
inputs

- these params are learned (componentwise) in training

---

### training w/ batch norm

take in a network and a subset of activations to transform and return a batch
normalized network, ready for inference

we transform the elements of inputs feature-wise across the batch

- not the entire activation vector

---

### CNNs w/ batch norm

for maintenance of the convolution property, normalize all activations in a mini
batch over all locations

normalization should be the same for spatially similar inputs.

mini-batch statistics are computer per feature map

#### method of normalization

inputs are normalized based on seeing the batch data so network is not fully
deterministic for training examples

### how to make bn work well

- increase learning rate
- remove dropout
- reduce l2 regularization
- speed up lr decay
- remove LRU
- shuffle batches better
- reduce distortions in data

### main takeaway

normalize inputs to layers using mini-batch stats --> speeds up training by
stabilizing distributions of incoming data into layers (no excess learning every
time)
