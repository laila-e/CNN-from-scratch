# Convolutional Neural Network (CNN) 
## Implementation from Scratch
### Overview
In this project I try to implement a CNN from scratch using numpy only.The objective of this project is to gain a deeper understanding of the inner workings of CNNs and to build a basic framework for image classification tasks.
### Dataset 
This CNN implementation uses the MNIST dataset for training and evaluation. The MNIST dataset is a collection of 28x28 grayscale images of handwritten digits (0-9), widely used for digit recognition tasks. It consists of a training set of 60,000 images and a test set of 10,000 images.
### Structure 
The project is organized into four main files, each responsible for a specific component of the Convolutional Neural Network (CNN) implementation:
#### Convolution.py
his file contains the Convolution class, which is responsible for performing the convolution operation and handling the backpropagation of the convolutional layer.
#### pooling.py
Within this file, you'll find the Pooling class, which manages the pooling operation and the corresponding backpropagation for the pooling layer.
#### optimizer.py 
this file contain the class where I implement Adam optimizer from scratch.
#### fullyconnected.py
The FullyConnected file consists of the FullyConnectedLayer class. This class is responsible for the fully connected layer, which predicts the class of the input image.
#### main.py
The Main file, or main.py, is the entry point for running and executing the CNN. It likely contains the code to instantiate the CNN model, load the dataset, train the model, and evaluate its performance.
### Author
Laila El Ouedeghyry
### References
<div>
  [Convolutional Neural Network From Scratch](https://medium.com/latinxinai/convolutional-neural-network-from-scratch-6b1c856e1c07)
  [Convolutional Neural Network](https://youtube.com/playlist?list=PLuhqtP7jdD8CD6rOWy20INGM44kULvrHu&si=NYxFK2h_NAMNoaOG)
</div>
