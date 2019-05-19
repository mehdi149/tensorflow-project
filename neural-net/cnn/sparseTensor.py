import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)



img = mnist.test.images[0]
print(img.shape)
img.resize((784,1))
print(img.shape)
import numpy as np
img = np.transpose(img)


print(img.shape)
