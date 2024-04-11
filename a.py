import numpy as np
import copy
from time import time


class Activation(object):
    def __relu(self, x):
        return np.where(x >= 0, x, 0)

    def __relu_derivative(self, a):
        return np.where(a >= 0, 1, 0)

    def __softmax(self, x):
        x_exponent = np.exp(x - np.max(x, axis = -1, keepdims = True))
        return x_exponent / np.sum(x_exponent, axis = -1, keepdims = True)

    def __init__(self, activation = 'relu'):

        if activation == 'relu':
            self.function = self.__relu
            self.function_derivative = self.__relu_derivative
        elif activation == 'softmax':
            self.function = self.__softmax


class Layer(object):

    def __init__(self, n_in, n_out, optimizer, activation = 'relu'):
        self.input = None
        self.output_before_activation = None
        self.output = None

        self.activation = Activation(activation).function
        if activation == 'softmax':
            self.activation_derivative = None
        else:
            self.activation_derivative = Activation(activation).function_derivative

        self.Weight = np.random.uniform(
            low = -np.sqrt(6 / (n_in + n_out)),
            high = np.sqrt(6 / (n_in + n_out)),
            size = (n_in, n_out)
        )
        self.bias = np.zeros(n_out,)

        self.optimizer_weight = copy.copy(optimizer)
        self.optimizer_bias = copy.copy(optimizer)

    def forward(self, input, train = True):

        self.input = input
        self.output_before_activation = np.dot(input, self.Weight) + self.bias
        self.output = self.activation(self.output_before_activation)

        return self.output

    def backward(self, delta):

        if self.activation_derivative:
            delta = delta * self.activation_derivative(self.output_before_activation)

        grad_weight = np.dot(self.input.T, delta)
        grad_bias = np.sum(delta, axis=0, keepdims=True)

        self.Weight = self.optimizer_weight.update(self.Weight, grad_weight)
        self.bias = self.optimizer_bias.update(self.bias, grad_bias)

        delta = np.dot(delta, self.Weight.T)

        return delta
    

class DropoutLayer(object):

    def __init__(self, drop_prob: float = 0.5):
        self.drop_prob = drop_prob
        self.mask = None

    def forward(self, X: np.ndarray, train: bool = True) -> np.ndarray:
        if train:
            self.mask = np.random.rand(*X.shape) >= self.drop_prob
            return X * self.mask
        else:
            return X * (1 - self.drop_prob)

    def backward(self, delta: np.ndarray) -> np.ndarray:
        return delta * self.mask if self.mask is not None else delta
    

class BatchNormalization(object):

    def __init__(self, gamma, beta, optimizer, momentum = 0.9):

        self.gamma = gamma
        self.beta = beta
        self.momentum = momentum
        self.mean = 0
        self.var = 1
        self.gamma_optimizer = copy.copy(optimizer)
        self.beta_optimizer = copy.copy(optimizer)

    def forward(self, X, train = True):

        if self.mean is None:
            self.mean = np.mean(X, axis = 0)
            self.var = np.var(X, axis = 0)

        if train:
            mean = np.mean(X, axis = 0)
            self.mean = self.momentum * self.mean + (1 - self.momentum) * mean
            var = np.var(X, axis = 0)
            self.var = self.momentum * self.var + (1 - self.momentum) * var
        else:
            mean = self.mean
            var = self.var

        self.X_minus_mean = X - mean
        self.std = np.sqrt(var + 1e-6)
        self.X_norm = self.X_minus_mean / self.std
        output = self.gamma * self.X_norm + self.beta

        return output

    def backward(self, delta):

        gamma_old = self.gamma

        gamma_grad = np.sum(delta * self.X_norm, axis = 0)
        beta_grad = np.sum(delta, axis = 0)

        self.gamma = self.gamma_optimizer.update(self.gamma, gamma_grad)
        self.beta = self.beta_optimizer.update(self.beta, beta_grad)

        dX_norm = delta * gamma_old
        dvar = np.sum(dX_norm * self.X_minus_mean, axis = 0) * (-0.5) * (self.var + 1e-6)**(-3/2)
        dmean = np.sum(dX_norm * (1/self.std), axis = 0) + dvar * (1/delta.shape[0]) * np.sum(-2 * self.X_minus_mean, axis = 0)
        delta = (dX_norm * (1/self.std)) + (dmean / delta.shape[0]) + (dvar * 2 / delta.shape[0] * self.X_minus_mean)

        return delta
    

class Optimizer(object):

    def __init__(self, lr = 0.001, momentum = 0.9, weight_decay: float = 1e-2):

        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.grad = None

    def update(self, weight, delta):

        if self.grad is None:
            self.grad = np.zeros(weight.shape)

        self.grad = self.momentum * self.grad + (1 - self.momentum) * delta
        weight = weight * (1 - self.weight_decay) - self.lr * self.grad

        return weight
    

class MLP(object):

    def __init__(self, n_in, n_out, layers, optimizer, activation, BN=False, Dropout=False, dropout_prob=None):

        self.layers = []
        self.activation = activation
        self.optimizer = optimizer
        self.lr = self.optimizer.lr
        self.n_out = n_out

        self.layers.append(Layer(n_in, layer[0], optimizer, activation[0]))
        if Dropout:
            self.layers.append(DropoutLayer(dropout_prob[0]))
        if BN:
            self.layers.append(BatchNormalization(np.ones((1, layer[0])), np.zeros((1, layer[0])), optimizer))

        for i in range(1, len(layer)):
            self.layers.append(Layer(layer[i-1], layer[i], optimizer, activation[i]))
            if Dropout:
                self.layers.append(DropoutLayer(dropout_prob[i]))
            if BN:
                self.layers.append(BatchNormalization(np.ones((1, layer[i])), np.zeros((1, layer[i])), optimizer))

        self.layers.append(Layer(layer[-1], n_out, optimizer, activation[-1]))

    def CE_loss(self, y, predict_y):

        y_onehot = np.eye(self.n_out)[y].reshape(-1, self.n_out)
        predict_y = np.clip(predict_y, 1e-15, 1 - 1e-15)
        loss = -np.sum(np.multiply(y_onehot, np.log(predict_y)))
        delta = predict_y - y_onehot
        return loss, delta

    def forward(self, input, train = True):

        output = input
        for layer in self.layers:
            output = layer.forward(output, train)
        return output

    def backward(self, delta):

        for layer in reversed(self.layers):
            delta = layer.backward(delta)

    def fit(self, X, y, epochs = 100, batch_size = 100, print_per = 50):

        loss_list = []
        accuracy_list = []

        for epoch in range(epochs):

            if epoch == int(epochs*1/3):
                self.lr = self.lr / 5
                self.optimizer = Optimizer(lr = self.lr)
            elif epoch == int(epochs*2/3):
                self.lr = self.lr / 5
                self.optimizer = Optimizer(lr = self.lr)

            loss_temp = 0
            predict_y_all_batch = []

            start = time()

            for index in np.arange(0, X.shape[0], batch_size):
                X_batch = X[index: min(index+batch_size, X.shape[0])]
                y_batch = y[index: min(index+batch_size, X.shape[0])]

                predict_y = self.forward(X_batch)

                loss, delta = self.CE_loss(y_batch, predict_y)

                self.backward(delta)

                loss_temp += loss
                predict_y_all_batch.extend(predict_y)

            predict_y_all_batch = np.array(predict_y_all_batch)

            loss_list.append(loss_temp / X.shape[0])
            predict_y = np.argmax(predict_y_all_batch, axis = 1).reshape(-1,1)
            accuracy = np.sum(predict_y == y, axis = 0) / X.shape[0]
            accuracy_list.append(accuracy)

            if (epoch + 1) % print_per == 0:
                print("Epoch: %d\tTime: %.2fs\tLoss: %.5f\tAccuracy: %.2f%%" % (epoch+1, time()-start, loss_list[-1], accuracy_list[-1]*100))

        return loss_list, accuracy_list

    def predict(self, X, y):

        predict_y = self.forward(X, train = False)

        loss, _ = self.CE_loss(y, predict_y)

        accuracy = np.sum(np.argmax(predict_y, axis = 1).reshape(-1,1) == y, axis = 0) / X.shape[0]

        print("Loss: %.5f\tAccuracy:%.2f%%" % (loss, accuracy*100))

