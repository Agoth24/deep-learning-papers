# RNN overview

RNNs are used to detect patterns within sequential data

RNNs send information back into themselves through hidden states representing
cumulative information about the sequence

they take into account all previous data up until the timestep

at each timestep...

- compute hidden state
- compute output state --> repeat the process for stacked layers one-by-one

--> with stacked RNNs, layers use previous layer outputs as x inputs.

### backprop through time

unfold the RNN to apply backpropagation

\*sum up all loss (chain rule backpropagation) terms across update steps for
each variable for gradient vector

numerical stability problems w/ gradients arise through powers of hidden state
weight matrices in BPTT

---

### LSTM units

a combatant rnn unit architecture against the vanishing/exploding gradient
problem

3 gates...

- output: read cell entries
- input: read data into cell
- forget: reset cell contents

(say more)

### deep rnns

just stack recurrent units together

the hidden state for a layer become the input for the following layer

output matrix uses the hidden state from final layer

### bidirectional rnns

use posterior information to incorporate the ability to look ahead in sequences

(say more)

### encoder-decoder & seq2se2

uses an encoder network & a decoder network to encode state and decode state.
--> both RNNs

encoder aggregates the 'context' into a vector using final hidden state

long context was a problem for encoders so attention was introduced

### attention

attention is focusing on parts of sequences/input to make inference about what
lies outside the focus point

attention allows for shortcuts from input to context vector

**KEY**: the context vector is a weighted (with alighment scores summng to 1)
sum of hidden states of the input sequence

alignment scores tell us how well pairs of input at different positions match

### transformers

incorporates attention w/ parallelization

encodes each item's position in sequence using encoder-decoder architecture

encoder has a self-attention & feedforward layer, decoder also uses same layers
but with an attention sub-layer between

\*multi-headed attention allows the model to attend to info in different
subspaces of representations at different positions

softmax layer at the end turns decoder output into a word (vector the size of
learned vocabulary from training set)

### pointer networks

extends seq2seq w/ attention by creating a sequence of pointers to elements of
input

can be used to solve combinatorically challenging optimization problems
