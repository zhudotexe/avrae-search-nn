import numpy as np


def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1.0 - x)


class BasicNN:
    def __init__(self, input_, validation, layer_sizes, num_outputs):
        self.input = input_
        self.weights = []

        # hidden layer weights
        for i, size in enumerate(layer_sizes):
            if i == 0:
                layer = np.random.rand(self.input.shape[1], size)
            else:
                layer = np.random.rand(self.weights[-1].shape[1], size)
            self.weights.append(layer)

        # output layer weight
        self.weights.append(np.random.rand(self.weights[-1].shape[1], num_outputs))

        # hidden layer states
        self.state = [input_]
        for weights in self.weights:
            self.state.append(np.zeros(weights.shape[1]))

        self.validation = validation

    def feedforward(self):
        for i, weights in enumerate(self.weights):
            if i == 0:
                self.state[i] = self.input
            result = sigmoid(np.dot(self.state[i], weights))

            # assign output to next layer
            # always 1 less weight than layers
            # final state = output
            self.state[i + 1] = result

    def backprop(self):
        weight_grads = [np.zeros(1)] * len(self.weights)
        # application of the chain rule to find derivative of the loss function with respect to weights2 and weights1
        last_grad = (2 * (self.state[-1] - self.validation))
        for i in range(-1, -len(self.weights) - 1, -1):
            weight_grads[i] = np.dot(self.state[i - 1].T, last_grad * sigmoid_derivative(self.state[i]))

        # update the weights with the derivative (slope) of the loss function
        for weight, grad in zip(self.weights, weight_grads):
            weight += grad

    def get_loss(self):
        return ((self.validation - self.state[-1]) ** 2).sum()


class ExampleNN:
    def __init__(self, x, y, layer_size, output_size):
        self.input = x
        self.weights1 = np.random.rand(self.input.shape[1], layer_size)
        self.weights2 = np.random.rand(layer_size, output_size)
        self.y = y
        self.output = np.zeros(self.y.shape)

    def feedforward(self):
        self.layer1 = sigmoid(np.dot(self.input, self.weights1))
        self.output = sigmoid(np.dot(self.layer1, self.weights2))

    def backprop(self):
        # application of the chain rule to find derivative of the loss function with respect to weights2 and weights1
        d_weights2 = np.dot(self.layer1.T, (2 * (self.y - self.output) * sigmoid_derivative(self.output)))
        d_weights1 = np.dot(self.input.T, (np.dot(2 * (self.y - self.output) * sigmoid_derivative(self.output),
                                                  self.weights2.T) * sigmoid_derivative(self.layer1)))

        # update the weights with the derivative (slope) of the loss function
        self.weights1 += d_weights1
        self.weights2 += d_weights2

    def sample(self, input_):
        l1 = sigmoid(np.dot(input_, self.weights1))
        return sigmoid(np.dot(l1, self.weights2))

    def get_loss(self):
        return ((self.y - self.output) ** 2).sum()


if __name__ == '__main__':
    # X = np.array([[0, 0, 1],
    #               [0, 1, 1],
    #               [1, 0, 1],
    #               [1, 1, 1]])
    # y = np.array([[0], [1], [1], [0]])
    # net = BasicNN(X, y, [4], 1)
    # nn = ExampleNN(X, y, np.copy(net.weights[0]), np.copy(net.weights[1]))
    #
    # for _ in range(1500):
    #     nn.feedforward()
    #     net.feedforward()
    #     nn.backprop()
    #     net.backprop()
    #     # print(nn.output)
    #     print(nn.get_loss())
    #     # print(net.state[-1])
    #     # print(net.get_loss())
    #     # input()
    #
    # print(nn.output)

    X = np.array()
    y = np.array()
    nn = ExampleNN(X, y, 64, 497)
