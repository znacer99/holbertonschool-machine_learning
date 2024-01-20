#!/usr/bin/env python3
"""
defines Neuron class: define single neuron performing binary
classification
"""

import numpy as np


class Neuron:
    """
    represents single neuron performing binary
    classification.
    class constructor:
        def __init__(self, nx)

    public instance attributes:
        w: the weighs vector for the neuron
        b: the bias for the neuron
        a: the activated output of the neuron
    """

    def __init__(self, nx):
        """
        class constructor

        parameters:
            nx [int]: the number of inputs features to the neuron
            if nx is not an integer, raise a TypeError.
            if nx is less than 1, raise a ValueError.
        """

        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.W = np.random.randn(1, nx)
        self.b = 0
        self.A = 0
