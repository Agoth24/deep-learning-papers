# Negative Sampling (word2vec)

skip gram model can be improved in quality and training speed through
subsampling frequent words

neural network encoded vectors contain _linguistic regularities_ (invariances
between vectors similar in certain ways)

word representations miss the representation of idioms that bypass word meanings
--> move to phrase based models

vector addition of word embeddings can be meaningful

### skip gram model

maximize the log probability of context words given one word over a sliding
window of a sequece.

- TOO EXPENSIVE & IMPRACTICAL
- approximate w/ hierarchical softmax

#### hierarchical softmax

evaluate $log_2(W)$ output nodes instead of W to get your prob. dist.
