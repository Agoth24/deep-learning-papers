# Adam (Adaptive Momentum Estimation)

**problem**: want to improve the iterative methods of stochastic optimization in
high dimensional parameter spaces

**what it does**: estimates params for first and second moments of gradients

adam extends on the benefits of rmsprop's adagrad fix

advantages

- param updates invariant to gradient rescaling
- bounded stepsizes
- works on sparse gradients
- anneals stepsizes

adam is similar to rmsprop in effectiveness on non-stationary objectives

- different gradients/loss functions at each step

### optimization algorithm

for every step of the loop:

- get the gradient
- update biased first and second moment estimates.
- correct the biased moment estimates
- update parameters

---

m_t holds the ema of past gradients v_t holds the ema of past **squared**
gradients

rmsprop estimates parameters using momentum on rescaled gradient.

- adam directly estimates using first and second moments

adagrad works well for sparse gradients.

- it's a special case of adam with ß_1 = 0

#### experiments

adam performs better than adagrad on logistic regression and similar to sgd w/
nesterov momentum

adam performs similar to adagrad with sparse features

adam is effective in deep cnns

- marginally better than sgd with momentum
