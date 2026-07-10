# Long Short-Term Memory

backprop on rnns takes a long time

lstm is a gradient-based method that truncates negligible gradients

**core idea**: one cell, input, output, & forget gates

the goal is to enforce constant error flow through hidden states

- HOW? **constant error carousel**

### the lstm unit

input gates learn when to allow new info into the cell

output gates learn what info should be emitted to the rest of the network

constant error carousel error flow is kept constant by **truncated BPTT** -->
only the internal state recurrence is carried backwards

- O(W) time and space complexity overall
