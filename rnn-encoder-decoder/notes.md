# RNN Encoder-Decoder (GRU)

**goal**: come up with a statistical machine translation model FOR THE PURPOSE
OF learning good representations of phrases

- an RNN encoder-decoder network

### core idea

encode a variable length sequence into a fixed length vector w/ an rnn and
decode fixed length vectors into variable length sequences w/ another rnn

---

### the encoder

an rnn that sequentially reads each symbol of the sequence vector and update
hidden state along the way

- finalized by an EOS symbol

### the decoder

another rnn that generates output through next token prediction based on hidden
state and previous output token

both components are jointly trained to maximize the average conditional log
likelihood of the pairs of training sequences

---

the rnn encoder decoder can be used for sequence generation by next token
prediction or scoring sequence pairs

### gated recurrent unit

adaptively remembers and forgets

- contains a reset gate and update gate.

update gate controls how much of the previous state to keep, reset gate controls
how much of the previous state forms the candidate state

$$\textbf{h}_t=\textbf{z}_t \odot \textbf{h}_{t-1} + (1-\textbf{z}_t) \odot \tilde{\textbf{h}}_{t}$$
hidden state is sum of hadamard products of how much to keep and how much to
udpate w/ previous state and candidate state respectively
