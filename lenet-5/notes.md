# LeNet-5

gradient based learning replaces the need for manual feature selection

automatic learning is way more efficient

gradient-based learning is based on the relative simplicity of minimizing smooth, continuous functions vs combinatorial, discrete functions
- minimize loss but watch for generalization

high-capacity sgd learning finds reasonable regions
- overparameterization and flat regions

### conv nets

feedforward nets have no invariance to translation or distortion
- inefficient computation

local structure is also ignored in fully connected networks

**cnns force local receptive fields**

three main ideas of cnn architectures
- local receptive fields
- shared weights
- spatial subsampling

shared weights help with shift invariance

pooling layers help with robustness to distortion

**we strengthen invariance to translation by downsampling and increasing depth of representation (# of channels)**

### LeNet-5

convolutional layers have sparse connections to break symmetry and force extraction of complementary feature combinations

pooling layers control computational complexity
- but cause loss of information

ends with a fully connected layer and a radial basis function layer
- RBF layer computes vector of distances between the FC layer and some target weight vector
	- LOWEST class output = **prediction**

