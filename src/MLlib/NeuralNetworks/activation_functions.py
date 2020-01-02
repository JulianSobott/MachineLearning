"""
:module: MLlib.NeuralNetworks.activation_functions
:synopsis: Activation functions for neurons
:author: Julian Sobott

public functions
----------------

.. autofunction:: reLu
.. autofunction:: sigmoid
.. autofunction:: step
.. autofunction:: linear



"""
import numpy as np


def reLu(values: np.ndarray) -> np.ndarray:
    return np.maximum(0, values)


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(np.negative(values)))


def step(values: np.ndarray) -> np.ndarray:
    return np.heaviside(values, 0.5)


def linear(values: np.ndarray) -> np.ndarray:
    return values
