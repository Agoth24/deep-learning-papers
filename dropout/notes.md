# Dropout

**problem**: high capacity neural networks can overfit badly

dropout is a regularization method that randomly drops hidden units during learning
- adds a multiplicative bernoulli mask for each unit

#### why does it work?

- supposedly prevents co adaptation of units in training so each unit is load-bearing

in the quest for generalization...
- training many models for ensembling is expensive
- need lots of training data that may not exist
DROPOUT WORKS AROUND THESE

dropout is a form of model averaging 2^n thinned models
- **scale down** the weights at test time by multiplying by keep rate

#### motivation

sexual reproduction requires sets of genes to work well with random sets of genes
- robust property

this reduces the risk of co-adaptation of genes
- free rider genes

*restricting co-adaptation makes sure each hidden unit isn't relying on others to correct its errors

---

dropout is a better regularizer than many common methods on staple benchmarks
- also efficient to train and run inference on

**dropout paired with max-norm** regularization is way better than plain dropout

dropout implicitly induces sparsity of activations in networks

**dropout has a sweet spot** for maximal improvements (in terms of data size)
- dataset too small, overfits even with dropout
- dataset too big, overfitting wasn't a problem to begin with

dropout creates a tradeoff between generalization error and training time
- longer to train than regular networks

