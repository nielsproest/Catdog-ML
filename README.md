## DM873 Project 1
The aim of this project is to create a neural network able to distinguish be-
tween cats and dogs.

In this project i tried different neural network configurations to attempt to improve my neural network accuracy.

Initially i based it on VGG-16, a well known neural network for these sort of tasks that generally works well. But i discovered that training such a big network was both computationally expensive, and my laptop didnt have alot of compute.

So i moved to reducing VGG-16 into a smaller network, twice, where i finally landed on a 72% accuracy with a smaller network, after having changed the traning parameters a couple of times, where i tried both changing the learning rate, and also trying L1 and L2 regularization.
