* Repository github url : https://github.com/jai-mr/
* Assignment Repository : https://github.com/jai-mr/Assignment-7
* Submitted by : Jaideep Rangnekar
* Registered email id : jaideepmr@gmail.com

The model has been executed for 50 epochs 
1. Depthwise Separable Convolution
2. Dilated Convolution
3. GAP
4. Accuracy Details :
    Accuracy of plane : 83 %
    Accuracy of   car : 91 %
    Accuracy of  bird : 77 %
    Accuracy of   cat : 67 %
    Accuracy of  deer : 83 %
    Accuracy of   dog : 80 %
    Accuracy of  frog : 90 %
    Accuracy of horse : 87 %
    Accuracy of  ship : 92 %
    Accuracy of truck : 92 %


# Jupyter Notebook File reference executed in Colab
[https://github.com/jai-mr/Assignment-7/blob/master/07_CodeFinal.ipynb](https://github.com/jai-mr/Assignment-7/blob/master/07_CodeFinal.ipynb)

# Training/Test - Loss & Accuracy Curve
[https://github.com/jai-mr/Assignment-7/blob/master/images/accuracy.png](https://github.com/jai-mr/Assignment-7/blob/master/images/accuracy.png)

# Test vs Train Accuracy
[https://github.com/jai-mr/Assignment-7/blob/master/images/testvtrain.png](https://github.com/jai-mr/Assignment-7/blob/master/images/testvtrain.png)

# Mis-Classified Images
[https://github.com/jai-mr/Assignment-7/blob/master/images/misclassification.png](https://github.com/jai-mr/Assignment-7/blob/master/images/misclassification.png)


## HOW DO WE INCREASE CHANNEL SIZE AFTER CONVOLUTION
### DEPTHWISE SEPARABLE CONVOLUTION
* In the regular 2D convolution performed over multiple input channels, the filter is as deep as the input and lets us freely mix channels to generate each element in the output. 
* Depthwise convolutions don't do that - each channel is kept separate - hence the name depthwise

### Dilated Convolution
* Dilated convolution is a way of increasing the receptive view (global view) of the network exponentiallya and linear parameter accretion. 
* With this purpose, it finds usage in applications
* Cares more about integrating the knowledge of the wider context with less cost

References:

[https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html)

[https://erogol.com/dilated-convolution/#:~:text=In%20simple%20terms%2C%20dilated%20convolution,4%20means%20skipping%203%20pixels.&text=The%20figure%20below%20shows%20dilated%20convolution%20on%202D%20data.](https://erogol.com/dilated-convolution/#:~:text=In%20simple%20terms%2C%20dilated%20convolution,4%20means%20skipping%203%20pixels.&text=The%20figure%20below%20shows%20dilated%20convolution%20on%202D%20data.)

[https://towardsdatascience.com/a-basic-introduction-to-separable-convolutions-b99ec3102728#:~:text=Unlike%20spatial%20separable%20convolutions%2C%20depthwise,factored%E2%80%9D%20into%20two%20smaller%20kernels.&text=The%20depthwise%20separable%20convolution%20is,number%20of%20channels%20%E2%80%94%20as%20well.](https://towardsdatascience.com/a-basic-introduction-to-separable-convolutions-b99ec3102728#:~:text=Unlike%20spatial%20separable%20convolutions%2C%20depthwise,factored%E2%80%9D%20into%20two%20smaller%20kernels.&text=The%20depthwise%20separable%20convolution%20is,number%20of%20channels%20%E2%80%94%20as%20well.)