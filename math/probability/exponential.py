#!/usr/bin/env python3

"""
Creates a class for exponential.
"""

class Exponential:
    """
    Class for exponential distribution
    """

    def __init__(self, data=None, lambtha=1.):
        self.lambtha = float(lambtha)
        if self.lambtha <= 0:
            raise ValyeError("lambtha must be a positive value")
        if data is None:
            if type(data) is not list:
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = float(sum(data)) / len(data)
