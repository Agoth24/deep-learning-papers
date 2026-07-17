# Attention

**problem**: fixed-length context vectors are problematic for encompassing long
sequences

**fix**: allow for a soft search over input context to decide weightings at
output timesteps

\*word prediction is based on the weighted context vectors & previously
generated words

### neural machine translation

learn a parametrized model that maximizes the conditional probability of a
parallel corpus of text

the attention mechanism is a novel extension to the RNN encoder-decoder sequence
to sequence architecture

### rnn encoder-decoders

compute hidden state vectors for each timestep of input (functions of current
input and past hidden state) --> compress the hidden states from the input into
a context vector (usually just final hidden state) --> trained for next-word
prediction using context and prebvious predictions

### aligning & translating

the encoder is a **bidirectional RNN**, decoder searches through the source
sentence.

each target word has a distinct context vector.

- context vectors depend on the entire sequence but focus on the parts around
  the i-th word

context vectors at output timesteps are weighted sums of input annotations

- weights are computed by softmax on alignment scores

the annotation weights encapsulate their importance and **implement an attention
mechanism** in our decoder

#### bidirectional rnns for annotations

the encoder has a forward rnn and backward rnn which compute hiddens states over
input sequences and vertically stacks them

- allows input hidden states to obtain focused meanings incorporating previous
  and later information
