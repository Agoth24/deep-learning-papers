# Effective Attention-Based NMT

there are two main attention mechanisms: GLOBAL and LOCAL

- local performs better and is computationally cheaper --> more tailored context

## neural machine translation

neural machine translation models the conditional probability of target sequence
given a source sequence.

- can be broken into log conditional probability where given previous targets and
  representation encoding

most NMT models use RNN decoders but have different architectures

- differ in computation of context & choice of hidden units

training objective is to minimize negative log likelihood over training corpus.

### attention models

global attention uses all encoder hidden states for softmax, local attention
only uses a subset.

- used in attaining context vectors $c_t$

#### global attention

compute softmax alignment vector (size = number of source timesteps)

- score functions can be MODULARIZED

#### local attention

inspired by soft vs hard attention --> focused on a small window of context,
DIFFERENTIABLE

- window size is selected empirically
- easier to train than global attention

### input-feeding

not optimal to make attention decisions independently at each timestep

- take past alignment information into account

**_input to timesteps are concatenation of attentional vectors and inputs_**

#### benefits of input feeding:

- notify the model of previous alignments
- deepen the network (horizontal & vertical)
