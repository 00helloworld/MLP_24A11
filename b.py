import numpy as np
import copy
from time import time

# Activation function class
class Activation(object):
    def __init__(self, activation='relu'):
        # Supported activation functions and their derivatives
        self.activations = {
            'relu': (self.relu, self.relu_derivative),
            'softmax': (self.softmax, None)
        }
        
        self.set_activation(activation)

    def set_activation(self, activation):
        # Set the activation function and its derivative
        if activation in self.activations:
            self.function, self.function_derivative = self.activations[activation]
        else:
            raise ValueError(f"Unknown activation function: {activation}")

    def relu(self, x):
        # ReLU activation function
        relu = np.where(x >= 0, x, 0)
        return relu

    def relu_derivative(self, x):
        # Derivative of ReLU activation function
        relu_de = np.where(x >= 0, 1, 0)
        return relu_de

    def softmax(self, x):
        # Softmax activation function
        x_exp = np.exp(x - np.max(x, axis=-1, keepdims=True))
        sm = x_exp / np.sum(x_exp, axis=-1, keepdims=True)
        return sm
    

# Layer class
class Layer(object):

    def __init__(self, n_input, n_output, optimizer, activation='relu'):
        self.input = None
        self.logit = None
        self.output = None

        # Initialize activation function and its derivative
        self.activation = Activation(activation).function
        self.activation_derivative = Activation(activation).function_derivative if Activation(activation).function_derivative else None

        # Initialize weights and biases
        self.Weight = self.initialize_weights(n_input, n_output)
        self.bias = np.zeros(n_output,)

        # Copy optimizer for weights and biases
        self.optimizer_weight = copy.copy(optimizer)
        self.optimizer_bias = copy.copy(optimizer)

    def initialize_weights(self, n_input, n_output):
        # Initialize weights using He initialization
        limit = np.sqrt(6 / (n_input + n_output))
        return np.random.uniform(low=-limit, high=limit, size=(n_input, n_output))

    def forward(self, input):
        # Forward pass through the layer
        self.input = input
        self.logit = np.dot(input, self.Weight) + self.bias
        self.output = self.activation(self.logit)
        return self.output

    def backward(self, delta):
        # Backward pass through the layer
        if self.activation_derivative:
            delta = delta * self.activation_derivative(self.logit)

        grad_weight = np.dot(self.input.T, delta)
        grad_bias = np.sum(delta, axis=0, keepdims=True)

        # Update weights and biases
        self.Weight = self.optimizer_weight.update(self.Weight, grad_weight)
        self.bias = self.optimizer_bias.update(self.bias, grad_bias)

        delta = np.dot(delta, self.Weight.T)

        return delta
    

# Dropout layer class
class DropoutLayer(object):

    def __init__(self, dropout=0.5):
        self.dropout = dropout
        self.mask = None

    def generate_mask(self, shape):
        # Generate dropout mask
        mask = np.random.rand(*shape) >= self.dropout
        return mask

    def apply_mask(self, X, mask):
        # Apply dropout mask
        return X * mask

    def forward(self, X, train=True):
        # Forward pass through dropout layer
        if train:
            self.mask = self.generate_mask(X.shape)
            return self.apply_mask(X, self.mask)
        else:
            return self.apply_mask(X, 1 - self.dropout)

    def backward(self, delta):
        # Backward pass through dropout layer
        update = delta * self.mask if self.mask is not None else delta
        return update
    

# Batch normalization class
class BatchNormalization(object):

    def __init__(self, gamma, beta, optimizer, momentum=0.95):
        self.gamma = gamma
        self.beta = beta
        self.momentum = momentum
        self.mean = None
        self.var = None

        # Copy optimizer for gamma and beta
        self.optimizer_gamma = copy.copy(optimizer)
        self.optimizer_beta = copy.copy(optimizer)

    def compute_mean_var(self, X):
        # Compute mean and variance
        mean = np.mean(X, axis=0)
        var = np.var(X, axis=0)
        return mean, var

    def update_mean_var(self, mean, var):
        # Update mean and variance using momentum
        if self.mean is None or self.var is None:
            self.mean = mean
            self.var = var
        else:
            self.mean = self.momentum * self.mean + (1 - self.momentum) * mean
            self.var = self.momentum * self.var + (1 - self.momentum) * var

    def normalize(self, X):
        # Normalize the input
        self.X_minus_mean = X - self.mean
        self.std = np.sqrt(self.var + 1e-6)
        self.X_norm = self.X_minus_mean / self.std

    def forward(self, X, train=True):
        # Forward pass through batch normalization
        if train:
            mean, var = self.compute_mean_var(X)
            self.update_mean_var(mean, var)
        else:
            mean = self.mean
            var = self.var

        self.normalize(X)
        output = self.gamma * self.X_norm + self.beta
        return output

    def compute_gradients(self, delta):
        # Compute gradients for gamma and beta
        grad_gamma = np.sum(delta * self.X_norm, axis=0)
        grad_beta = np.sum(delta, axis=0)

        dX_norm = delta * self.gamma
        dvar = np.sum(dX_norm * self.X_minus_mean, axis=0) * (-0.5) * (self.var + 1e-6)**(-3/2)
        dmean = np.sum(dX_norm * (1/self.std), axis=0) + dvar * (1/delta.shape[0]) * np.sum(-2 * self.X_minus_mean, axis=0)
        delta = (dX_norm * (1/self.std)) + (dmean / delta.shape[0]) + (dvar * 2 / delta.shape[0] * self.X_minus_mean)

        return grad_gamma, grad_beta, delta

    def backward(self, delta):
        # Backward pass through batch normalization
        gamma_grad, beta_grad, delta = self.compute_gradients(delta)

        self.gamma = self.optimizer_gamma.update(self.gamma, gamma_grad)
        self.beta = self.optimizer_beta.update(self.beta, beta_grad)

        return delta
    

# Optimizer class
class Optimizer(object):

    def __init__(self, lr=0.001, momentum=0.95, weight_decay=1e-2):
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.velocity = None

    def initialize_velocity(self, shape):
        # Initialize velocity for momentum
        self.velocity = np.zeros(shape)

    def compute_velocity(self, delta):
        # Compute velocity for momentum
        self.velocity = self.momentum * self.velocity + (1 - self.momentum) * delta

    def update_weight(self, weight):
        # Update weights with weight decay and momentum
        weight = weight * (1 - self.weight_decay) - self.lr * self.velocity
        return weight

    def update(self, weight, delta):
        # Update weights using optimizer
        if self.velocity is None:
            self.initialize_velocity(weight.shape)
        
        self.compute_velocity(delta)
        weight = self.update_weight(weight)

        return weight
    

# MLP class
class MLP(object):

    def __init__(self, n_input, n_output, hidden_layers, optimizer, activation, BN=False, Dropout=False, dropout=None):

        self.layers = []
        self.activation = activation
        self.optimizer = optimizer
        self.lr = self.optimizer.lr
        self.n_out = n_output

        # Add layers to the model
        self.add_layers(n_input, hidden_layers, BN, Dropout, dropout)

    def add_layers(self, n_input, hidden_layers, BN, Dropout, dropout):
        # Add layers with optional Batch Normalization and Dropout
        for i, (layer_size, activation) in enumerate(zip(hidden_layers, self.activation)):
            self.layers.append(Layer(n_input if i == 0 else hidden_layers[i-1], layer_size, self.optimizer, activation))
            
            if Dropout:
                self.layers.append(DropoutLayer(dropout[i]))
            if BN:
                self.layers.append(BatchNormalization(np.ones((1, layer_size)), np.zeros((1, layer_size)), self.optimizer))

        self.layers.append(Layer(hidden_layers[-1], self.n_out, self.optimizer, self.activation[-1]))

    def CE_loss(self, y, predict_y):
        # Compute cross-entropy loss and its gradient
        y_onehot = np.eye(self.n_out)[y].reshape(-1, self.n_out)
        predict_y = np.clip(predict_y, 1e-15, 1 - 1e-15)
        loss = -np.sum(np.multiply(y_onehot, np.log(predict_y)))
        delta = predict_y - y_onehot
        return loss, delta

    def forward(self, input):
        # Forward pass through the network
        output = input
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def backward(self, delta):
        # Backward pass through the network
        for layer in reversed(self.layers):
            delta = layer.backward(delta)

    def fit(self, X, y, epochs=100, batch_size=100, print_per=50):
        # Training method
        loss_list = []
        accuracy_list = []

        for epoch in range(epochs):
            self.update_lr(epoch, epochs)

            loss_sum = 0
            predict_y_batch = []

            start = time()

            for i in np.arange(0, X.shape[0], batch_size):
                X_batch = X[i: min(i+batch_size, X.shape[0])]
                y_batch = y[i: min(i+batch_size, X.shape[0])]

                predict_y = self.forward(X_batch)

                loss, delta = self.CE_loss(y_batch, predict_y)

                self.backward(delta)

                loss_sum += loss
                predict_y_batch.extend(predict_y)

            predict_y_batch = np.array(predict_y_batch)

            loss_list.append(loss_sum / X.shape[0])
            predict_y = np.argmax(predict_y_batch, axis=1).reshape(-1, 1)
            accuracy = np.sum(predict_y == y, axis=0) / X.shape[0]
            accuracy_list.append(accuracy)

            if (epoch + 1) % print_per == 0:
                print("Epoch: %d\tTime: %.2fs\tLoss: %.5f\tAccuracy: %.2f%%" % (epoch+1, time()-start, loss_list[-1], accuracy_list[-1]*100))

        return loss_list, accuracy_list

    def update_lr(self, epoch, total_epochs):
        # Update learning rate at specific epochs
        if epoch == int(total_epochs*1/3) or epoch == int(total_epochs*2/3):
            self.lr /= 5
            self.optimizer = Optimizer(lr=self.lr)

    def predict(self, X, y):
        # Prediction method
        predict_y = self.forward(X)

        loss, _ = self.CE_loss(y, predict_y)
        accuracy = np.sum(np.argmax(predict_y, axis=1).reshape(-1, 1) == y, axis=0) / X.shape[0]

        print("Loss: %.5f\tAccuracy:%.2f%%" % (loss, accuracy*100))


if __name__ == '__main__':
    from sklearn.preprocessing import MinMaxScaler
    # Load data
    X_train = np.load('train_data.npy')
    X_test = np.load('test_data.npy')
    y_train = np.load('train_label.npy')
    y_test = np.load('test_label.npy')
    
    scaler = MinMaxScaler()
    train_data_scaled = scaler.fit_transform(X_train)
    test_data_scaled = scaler.transform(X_test)

    # Initialize optimizer
    optimizer = Optimizer(lr=0.001, momentum=0.8, weight_decay=1e-3)

    n_input = X_train.shape[1]
    n_output = len(np.unique(y_train))

    layer2 = [256, 512, 256]

    activation = ['relu', 'relu', 'relu', 'softmax']
    model = MLP(n_input, n_output, layer2, optimizer, activation, BN=True, Dropout=True, dropout=[0.2, 0.2, 0.2])

    # Train the model
    loss, accuracy = model.fit(X_train, y_train, epochs=30, batch_size=1000, print_per=3)

    print()

    # Test the model
    model.predict(X_test, y_test)
