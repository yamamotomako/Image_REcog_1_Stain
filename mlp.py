import tensorflow as tf
import tensorflow.contrib.keras as keras
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import sys

import matplotlib.pyplot as plt
import keras
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout

num_classes = 10
im_rows = 32
im_cols = 32
im_size = im_rows*im_cols*3

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.reshape(-1, im_size).astype("float32")/255
x_test = x_test.reshape(-1, im_size).astype("float32")/255
y_train = y_train.reshape(-1, im_size).astype("float32")/255
y_test = y_test.reshape(-1, im_size).astype("float32")/255

model = Sequential()
model.add(Dense(512, activation="relu", input_shape=(im_size,)))
model.add(Dense(num_classes, activation="softmax"))

model.compile(
    loss="categorical_crossentropy",
    optimizer="adam",
    metrisc=["accuracy"]
)

hist = model.fit(x_train, y_train,
    batch_size=32, epochs=10,
    verbose=1,
    validation_data=(x_test, y_test)
)

score = model.evaluate(x_test, y_test, verbose=1)
print score[0], "loss=",score[1]

plt.plot(hist.history["acc"])
plt.plot(hist.history["val_acc"])
plt.show()

plt.plot(hist.history["loss"])
plt.plot(hist.history["val_loss"])
plt.show()




sys.exit()




iris_data = pd.read_csv("./iris.csv", encoding="utf-8")
print type(iris_data)



y_labels = iris_data.loc[:,"Name"]
x_data = iris_data.loc[:, ["SepalLength","SepalWidth","PetalLength","PetalWidth"]]



labels = {
    "Iris-setosa": [1,0,0],
    "Iris-versicolor": [0,1,0],
    "Iris-virginica": [0,0,1]
}

print y_labels


y_nums = np.array(list(map(lambda v : labels[v], y_labels)))
x_data = np.array(x_data)

print y_nums

x_train, x_test, y_train, y_test = train_test_split(x_data, y_nums, train_size=0.8)


#keras
Dense = keras.layers.Dense
model = keras.models.Sequential()
model.add(Dense(10, activation="relu", input_shape=(4,)))
model.add(Dense(3, activation="softmax"))

model.compile(
    loss = "categorical_crossentropy",
    optimizer = "adam",
    metrics=["accuracy"]
)

model.fit(x_train, y_train, batch_size=2-0, epochs=300)

score = model.evaluate(x_test, y_test, verbose=1)
print score[1], "loss="+score[0]





#tensorflow
#x  = tf.placeholder(tf.float32, [None, 4])
#y_ = tf.placeholder(tf.float32, [None, 3])

#w = tf.Variable(tf.zeros([4,3]))
#b = tf.Variable(tf.zeros([3]))

#y = tf.nn.softmax(tf.matmul(x,w)+b)

#cross_entropy = -tf.reduce_sum(y_ * tf.log(y))





sys.exit()




a = tf.constant(10, name="10")
b = tf.constant(20, name="20")
c = tf.constant(30, name="30")

add_op = tf.add(a, b, name="add")
mul_op = tf.multiply(add_op, c, name="c")

sess = tf.Session()
res = sess.run(mul_op)
print res

tf.summary.FileWriter("./log.txt", sess.graph)








