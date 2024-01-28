#!/ur/bin/env python3
"""
defines a Neuron class that defines a single neuron performing binary
classification based on 0-neuron.py
"""


import numpy as np


class Neuron:
    """
    class that represents a single neuron performing bianry
    classification

    class constructor:
        def __init__(self, nx)

    private instance attributes:
        _W: the weights vector for the neuron
        _b: the bias for the neuron
        _A: the activated output of the neuron (prediction)
    """


    def __init__(self, nx):
        """
        class constructor

        parameters:
            nx [int]: the number of input features to the neuron
            if nx is not an integer, raise a TypeError.
            if nx is less than 1, raise a ValueError.

        sets private instance attributes:
            __W: the weights vector for the neuron,
                initialized using a random normal distribution
            __b: the bias for the neuron,
                initialized to 0
            __A: the activated output for the neuron (prediction),
                initialized to 0
        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.__w = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """
        getter method
        args:
            - self
        return:
        - __W: the value of the proate attribute __W.
        """
        return self.__W

    @property
    def A(self):
        """
        getter method
        Args:
            - self
        Return:
        - __A: The value of the proate attribute __A.
        """
        return self.__A

    @property
    def b(self):
        """
        getter method
        Args:
            - self
        Return:
        - __b: The value of the proate attribute __b.
        """
        return self.__b
