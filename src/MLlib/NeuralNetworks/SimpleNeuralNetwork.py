"""
:module: MLlib.NeuralNetworks.SimpleNeuralNetwork
:synopsis: Simple and easy to understand NN class
:author: Julian Sobott

public classes
---------------

.. autoclass:: SimpleNeuralNetwork
    :members:


"""
from typing import List
import numpy as np

from MLlib.NeuralNetworks.activation_functions import reLu


class SimpleNeuralNetwork:

    def __init__(self, layers: List[int], activation_functions=None):
        np.random.seed(100)
        self.layers = layers
        self.neurons = [np.zeros(size) for size in self.layers]
        self.weights = [np.random.rand(in_layer, out_layer)
                        for in_layer, out_layer in zip(self.layers[:-1], self.layers[1:])
                        ]
        self.activation_functions = [None, *activation_functions] if activation_functions else \
            [reLu for _ in self.layers]

    def calc_output(self, in_values: np.ndarray):
        assert in_values.shape == self.neurons[0].shape, f"In values don't match NN input shape: {in_values.shape} vs. " \
                                                   f"{self.neurons[0].shape}"
        self.neurons[0] = in_values
        for layer in range(1, len(self.layers)):
            self.neurons[layer] = np.dot(self.neurons[layer - 1], self.weights[layer - 1])
            self.neurons[layer] = self.activation_functions[layer](self.neurons[layer])
        return self.neurons[-1]

    def copy(self):
        import copy
        copy_nn = SimpleNeuralNetwork(self.layers[:], self.activation_functions[:])
        copy_nn.activation_functions = self.activation_functions[:]
        copy_nn.neurons = copy.deepcopy(self.neurons)
        copy_nn.weights = copy.deepcopy(self.weights)
        return copy_nn


if __name__ == '__main__':
    nn = SimpleNeuralNetwork(layers=[2, 3, 1])
    out = nn.calc_output(np.array([2, 2]))
    print(out)  # out: [2.47912808], seed: 100
