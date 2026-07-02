# AlexNet

a large conv net architecture that performed well on the 1000 class imagenet
challenge

5 conv layers followed by 3 fully connected
- uses dropout & max pooling

larger training datasets are required to match real life variablity

- but we need high model capacity to keep up

large dataset image classification still requires some priors on our data

- convolutional networks supply some useful priors

#### main contributions

achieved the best performance on imagenet datasets

novel features in the model, improving training time and accuracy

convolutional layers are orders of magnitude more crucial for image
classification than other types of layers

### model architecture

8 layers: 5 convolutional, 3 fully connected. ends with 1000-way softmax

uses non-saturating neurons for training efficiency (ReLU)

- learns way faster than saturating neurons like tanh and sigmoid

alexnet was trained on two parallelized GPUs because of the large training set

predictions are made by maximing multinomial log-reg objective --> softmax + NLL
(or categorical cross entropy).

### regularization methods

employed label-preserving data augmentation on the dataset

- increased training set size by a factor of 2048

performed PCA on the RGB values

- to capture property of invariance of color intensity

uses dropout as a more efficient alternative to model ensembling

- super crucial to prevent overfitting
- doubles number of iterations to convergence

#### learning/optimization

sgd w/ momentum and weight decay to help converge better

weights initialized from zero mean gaussian w/ variance 0.01

- biases in all but first and third layer init w/ 1

---

feature vectors with small euclidean distance are similar in higher layers of
the network
