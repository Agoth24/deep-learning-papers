**goal**: cheaply learn and compute good dense word embeddings that capture semantic meaning
___

need to move away from atomic word models like N-gram because too basic of a technique

similar word vectors should be close to each other and can have varying degrees of similarity

computational complexity for continuous word representation models are proportional to: $$O=E \times T \times Q$$
- epochs x training words x Q defined per model

### continuous bag of words

average the surrounding context words and classify the current words: $$Q = N \times D + D \times log_2(V)$$
### continuous skip-gram

predict surrounding words based on current word: $$Q = C \times (D + D \times log_2(V))$$
c is the *max distance* of words

---
cbow faster to train and 

### summary

can cheaply train high quality word vectors w/ relatively simple models

the word vectors now capture syntactic and semantic regularities

computational complexity is cheap so we can compute high dimensional vector representations from large datasets w/ accuracy