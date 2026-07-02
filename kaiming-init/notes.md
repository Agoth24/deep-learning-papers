# Kaiming Init

provides a more robust parameter initialization scheme for ReLU networks

ReLU speeds up convergence and gets to better solutions

PReLU allows for learnable parameters in rectifiers for minimal computational cost
- allows for slight improvements over ReLU

### initialization of kernel weights

previously, deep cnns we're initialized by zero mean gaussians

Xavier init's linear activation assumption is INVALID for ReLU.
- we need another initialization scheme

kaiming init initializes weights from a zero-mean gaussian distribution w/ variance $\frac{2}{n_l}$

---

Xavier init does not converge for deep ReLU networks due to half of preactivations getting zeroed (zero-mean init)
- Kaiming init converges due to doubling the weight variance from the halving done by ReLU
	- gives us the propagation factor = 1 for forward and backward passes