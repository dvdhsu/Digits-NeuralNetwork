# MNIST Digits Neural Network

This neural network contains two hidden layers, trains on on [Yann LeCun's MNIST training set](http://yann.lecun.com/exdb/mnist/), and scores reasonably (97.49 with the provided weights) on on [Yann LeCun's MNIST test set](http://yann.lecun.com/exdb/mnist/). To train the neural network yourself, run digits.m in Octave. There are a few parameters you can set:

* Hidden layer dimensions: Default is 410 and 100; this can be changed in digits.m, line 9 and 10. 

* Number of iterations: Default is 500, which can take a long time. This can be changed in digits.m, line 53.

* Regularization parameter (lambda): Default is 6. This can be changed in digits.m, line 56. 

Once the training is complete, the test set accuracy will be calculated and printed. After the program has finished, you can save your weights by running the command "save [filename] Theta1 Theta2 Theta3". To load the saved data, run the command "load [filename]". 

Much thanks to [Sam Roweis](http://www.cs.nyu.edu/~roweis/) for the [data](http://www.cs.nyu.edu/~roweis/data.html) in MATLAB format. 

