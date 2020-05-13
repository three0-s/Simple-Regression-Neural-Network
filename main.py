# 2020.05.06     Regression of a function f = 4x^2+ 15 using DNN without Deep Learing Framework
import numpy as np
from common.layers import *
from common.functions import *
from common.optimizer import *
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# error setting
class numbering_err(Exception):
    def __str__(self):
        return "<i_layers> and the length of <t_shapes> should be same"


# class declaration
class SRNN:
    """ i_layers means the number of the affine layers,
        t_shapes refers to the number of neurons of each layer.
        Prefix reveals the data type of each variable."""

    def __init__(self, i_layers, t_shapes, s_optimizer="Adam"):
        self.params = {}
        self.layers = {}

        if s_optimizer == "Adam":
            self.optimizer = Adam()
        elif s_optimizer == "SGD":
            self.optimizer = SGD()

        try:
            if not i_layers == len(t_shapes) - 1:
                raise numbering_err()

        except numbering_err as e:
            print(e)

        for i in range(i_layers):
            self.params["w" + str(i + 1)] = np.random.randn(t_shapes[i], t_shapes[i + 1]) / np.sqrt(t_shapes[i] / 2)
            self.params["b" + str(i + 1)] = np.zeros(t_shapes[i + 1])

            self.layers["Affine" + str(i + 1)] = Affine(self.params["w" + str(i + 1)], self.params["b" + str(i + 1)])
            self.layers["ReLU" + str(i + 1)] = Relu()

        del self.layers["ReLU" + str(i_layers)]
        self.layers["Identity_with_loss"] = Identity_with_loss()

    def predict(self, x):
        x_buf = x

        for layer in self.layers.keys():
            if layer == "Identity_with_loss":
                pass
            else:
                x_buf = self.layers[layer].forward(x_buf)

        return identity_function(x_buf)

    def loss(self, x, t):
        y = self.predict(x)
        err = self.layers["Identity_with_loss"].forward(y, t)
        return err

    def gradient(self, x, t):
        dout = 1
        self.loss(x, t)

        for layer in list(reversed(list(self.layers.keys()))):
            dout = self.layers[layer].backward(dout)

        grads = {}
        affine_num = int(len(list(self.layers.keys())) / 2)

        for i in range(affine_num):
            grads["w" + str(i + 1)] = self.layers["Affine" + str(i + 1)].dW
            grads["b" + str(i + 1)] = self.layers["Affine" + str(i + 1)].db

        return grads

    def fit(self, x, t, batch_size, epoch=100):
        iteration = int(x.shape[0] / batch_size)

        for i in range(epoch):
            batch_mask = np.random.choice(x.shape[0], batch_size)
            x_train = x[batch_mask]
            t_train = t[batch_mask]

            for j in range(iteration):
                grads = self.gradient(x_train, t_train)
                self.optimizer.update(self.params, grads)

            print(str(i + 1) + "/" + str(epoch) + " epoch ...")

        print("Learning process has been done!")

# An example of regression using SRNN

network = SRNN(5, (1, 32, 64, 128, 64, 1), s_optimizer="Adam")
x = np.arange(-15, 15, 0.1)[np.newaxis].T
y = 4 * (x ** 2) + 15

x_train = x[0:240]
y_train = y[0:240]
network.fit(x_train, y_train, 10)

x_test = x[240:]
y_test = y[240:]
pre1 = network.predict(x_test)

score = r2_score(y_test, pre1)
print("test score: " + str(score))

pre2 = network.predict(x_train)
score = r2_score(y_train, pre2)
print("train score: " + str(score))

plt.figure(figsize=(12, 4))
plt.scatter(x_test, y_test, alpha=0.7, label='test_true')
plt.scatter(x_test, pre1, alpha=0.7, label='test_pred')
plt.scatter(x_train, y_train, alpha=0.5, label='train_true')
plt.scatter(x_train, pre2, alpha=0.5, label='train_pred')

plt.legend()

# This model has an overfitting problem. Find out how to avoid this.
plt.savefig('4x^2+15_regression_with_overfitting.png')
plt.show()
