# VGGNet

an investigation into the importance of depth in the image recognition accuracy of convolutional neural networks

extending depth significantly improves conv nets using small kernels

essentially an architecture tweak to previous conv nets on image recognition

### model architecture

3 x 224 x 224 input to the model, uses 3x3 kernels throughout
- smallest possible kernel size w/ spatial notions

convolutional layers followed by 3 fully connected (4096-4096-1000 channels) then softmax
- ReLU after each layer

---

the idea of using more small kernels rather than fewer larger ones is to get more nonlinearity in the decision function & less params

### training & testing

trained by optimizing softmax + NLL objective w/ momentum
- mini batch sgd, weight decay, dropout

deep nets with small filters outperform shallow nets with larger filters

### main takeaway

stacking small kernels in subsequent layers achieves better performance in conv nets compared to fewer, larger kernels with the same receptive field size
- e.g., 3x3 vs 11x11 kernels