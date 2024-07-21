#!/usr/bin/env python3

"""
Creates a class for exponential.
"""


class Exponential:
    """
    Class for exponential distribution
    """

    def __init__(self, data=None, lambtha=1.):
        if data is None:
            if lambtha < 1:
                raise ValueError("lambtha must be a positive value")
            else:
                self.lambtha = float(lambtha)

        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            elif len(data) < 2:
                raise ValueError("data must contain multiple values")
            else:
                lambtha = float(len(data)) / sum(data)
                self.lambtha = lambtha

    def pdf(self, x):
        """
        Calculates the value of the PDF for a given time period
        Parameters:
            x [int]: time period
                if x is out of range, return 0

        return:
            the PDF value for x
        """
        if x < 0:
            return 0
        e = 2.7182818285
        lambtha = self.lambtha
        pdf = lambtha * (e ** (-lambtha * x))
        return pdf

    def cdf(self, x):
        """
        Calculates the value of the CDF for a given time period
        return:
            the CDF value for x
        """
        if x < 0:
            return 0
        e = 2.7182818285
        lambtha = self.lambtha
        cdf = 1 - (e ** (-lambtha * x))
        return cdf
