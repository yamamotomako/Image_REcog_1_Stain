import numpy as np
import mxnet as mx
import logging

x = np.asarray([[0,0],[0,1],[1,0],[1,1]], dtype=np.float32)
y = np.asarray([0,1,1,0], dtype=np.int32)
batch_size=4

logging.basicConfig(level=logging.DEBUG, format='%(asctime)-15s %(message)s')

net = mx.symbol.Variable("data")
net = mx.symbol.FullyConnected(data=net, name="fc1", num_hidden=2)
net = mx.symbol.Activation(data=net, name="sigmoid1", act_type="sigmoid")
net = mx.symbol.FullyConnetcted(data=net, name="fc2", num_hidden=2)
net = mx.symbol.SoftmaxOutput(data=net, name="softmax")

model = mx.model.FeedForward(
    ctx = mx.cpu(),
    symbol = net,
    numpy_batch_size = batch_size,
    num_epoch = 10,
    learning_rate = 1,
    momentum = 0.9,
    initializer      = mx.init.Xavier(factor_type='in')
)

model.fit(
    x = x,
    y = y,
    eval_data = (x,y),
    eval_metric = ["accuracy"]
)


print model.predict(x)






