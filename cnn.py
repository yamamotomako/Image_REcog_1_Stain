#! /usr/bin/env python

import os,sys
import cnn_model
import keras
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

im_rows = 150
im_cols = 250
im_color = 3
in_shape = (im_rows, im_cols, im_color)
nb_classes = 2

photos = np.load("./all.npz")
x = photos["x"]
y = photos["y"]

x = x.reshape(-1, im_rows, im_cols, im_color)
x = x.astype("float32")/255

y = keras.utils.np_utils.to_categorical(y.astype("int32"), nb_classes)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8)

model = cnn_model.get_model(in_shape, nb_classes)

hist = model.fit(x_train, y_train,
    batch_size = 32,
    epochs = 40,
    verbose = 1,
    validation_data = (x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=1)
print "accuracy: ", score[1], "loss="+str(score[0])

plt.plot(hist.history["acc"])
plt.plot(hist.history["val_acc"])
plt.title("Accuracy")
plt.legend(["train", "test"], loc="upper left")
plt.show()

model.save_weights("./photos-model-light.hdf5")





