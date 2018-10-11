import os
import sys

os.environ["KERAS_BACKEND"] = "theano"

import keras as ks
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.utils import np_utils

from sklearn.datasets import fetch_mldata

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


mnist = fetch_mldata("MNIST original")
x, y = mnist["data"], mnist["target"]

print x.shape
print y.shape

test_number = x[53238]
test_number_image = test_number.reshape(28, 28)
pd.options.display.max_columns = 28
number_matrix = pd.DataFrame(test_number_image)

plt.imshow(test_number_image, cmap = matplotlib.cm.binary,
           interpolation='nearest')
plt.show()



print number_matrix
print y[53238]

x_train, x_test, y_train, y_test = x[:60000], x[60000:], y[:60000], y[60000:]
y_test_bkup = y_test

print (x_train, x_test)
print (y_train, y_test)

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
x_train = x_train.astype("float32")
x_test = x_test,astype("float32")

x_train /= 255
x_test /= 255

y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)

print y_test[1]



model = Sequential()
model.add(Dense(32, input_dim=784))
model.add(Activation("relu"))












