# Negative Sampling (word2vec)

**problem:** skip-gram over entire vocabularies is too expensive

- can be improved in quality and training speed through subsampling frequent
  words

neural network encoded vectors contain _linguistic regularities_ (invariances
between vectors similar in certain ways)

word representations miss the representation of idioms that bypass word meanings
--> move to **phrase based models**

vector addition of word embeddings can be meaningful

### skip gram model

maximize the log probability of context words given one word over a sliding
window of a sequece.

- TOO EXPENSIVE & IMPRACTICAL
- approximate w/ hierarchical softmax

#### hierarchical softmax

evaluate $log_2(W)$ output nodes instead of W to get your prob. dist.

- $log_2$ because of binary tree representation of vocabulary

### negative sampling

noise contrastive estimation: models should differentiate data from noise by
logistic regression (binary classification)

new (more efficient) task is to distinguish target words from noise words using
logistic regression w/ k negative samples for each data point

#### frequent-word subsampling

used to disentangle target words from frequent words like "the"

**IDEA**: discard words relative to their frequency, controlled by a
hyperparameter

speeds up training time.

### phrase learning

don't expand the training vocab for phrases, use the words within the vocab

form phrases by scoring unigram and bigram counts and choosing those scoring
above some threshold

### additive compositionality

another linear property of word vectors is the ability to add words and get
another word with the meaning of the composition

- equivalent to adding the log probabilities of the softmax layer
    - THE PRODUCT OF THE TWO CONTEXT DISTRIBUTIONS

works because word embeddings represent the distributions of the word context
