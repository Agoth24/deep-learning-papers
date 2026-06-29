# Deep Sparse Rectifier Networks (ReLU)

rectifying neurons are more representative of biological neurons

empirical results show that models train better when neurons are either off or linear

rectifier units turn off roughly half of the neurons so resulting networks are sparse.
- similar to biology

---
### neursoscience tie-in

evidence suggests neurons encode information sparsely
- 1-4% of neurons simultaneously active

sigmoid just doesn't make sense with steady state of 1/2 for activations
- not biologically realistic
- bad for gradient optimization

sparsity is good for:
- disentangling information
- variable-size representations
- being linear separable
- distributed

---

### deep rectifier networks

real life neurons are rarely in saturation regime
- activation function of neurons are more like rectifiers

#### what is it good for?

- allows the network to learn sparse representations by default
- just a bunch of linear models
	- good for computation
- allows smooth gradient flow through active neurons

---
### summary

sparsity & linear activations are good for deep networks.

rectifiers help finding good minima during optimization
- (simplifies gradients)

ill-suited activation functions cause difficulty in training deep networks becuase of lost gradients

the problem with training deep networks was never initialization, it was imprecise activation functions

