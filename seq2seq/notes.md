# Sequence-to-Sequence Learning (seq2seq)

**question**: how do you apply deep neural nets to map sequences to sequences?

seq2seq is a DNN approach to sequence learning that learns invariant phrase and
sentence representions.

- a generalization of Cho et al.'s phrase learning

### why don't DNNs work on sequences?

because the supervised learning technique in neural nets only works for
fixed-length vectors

- speech and langauge is NOT fixed length
- neither is dialogue

### core idea

use an lstm to process a variable-length sequence and encode a fixed dimension
context vector --> use another lstm to extract output sequence from context

### the seq2seq model

the goal of the LSTM is to estimate conditional probability of an output
sequence given an input sequence

with an lstm, encode the input sequence using the final hidden state into a
fixed-length vector representation --> decode it with another lstm

training objective: maximize log conditional probability of a correct
translation sentence given a source sentence

### why reverse the source sequence?

minimize the time lag between source & target for early outputs --> stabilize
the communication

by PCA projections, **the model learns underlying sentence representations**
regardless of active/passive voice and word ordering
