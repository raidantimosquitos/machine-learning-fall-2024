# Machine learning model to classify the MNIST dataset

This assignment presents a supervised learning classifier model that will try to solve the task of classifying handwritten digits from 0 to 9 in the [MNIST dataset](https://en.wikipedia.org/wiki/MNIST_database). The approach followed is to feed all the images through two models parallely: a Convolutional Neural Network (CNN) and a Long Short Term Memory (LSTM) neural network. The otuput of these two neural networks is concatenated and fed through a Fully Connected Layer (FCL) that will perform the classification. An illustration of the architecture can be found below:



