#!/usr/bin/env python3
"""
Defines Binomial class that represents a binomial distribution
"""


class Binomial:
    """
    class that represent binomial ditribution
    """
    def __init__(self, data=None, n=1, p=0.5):
        if data is None:
            if n < 1:
                raise ValueError("n must be a positive value")
            else:
                self.n = n
            if p <= 0 or p >= 1:
                raise ValueError("p must be greater than 0 and less than 1")
            else:
                self.p = p
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            elif len(data) < 2:
                raise ValueError("data must contain multiple values")
            else:
                mean = float(sum(data) / len(data))
                summation = 0
                for x in data:
                    summation += ((x - mean) ** 2)
                variance = (summation / len(data))
                q = variance / mean
                p = (1 - q)
                n = round(mean / p)
                p = float(mean / n)
                self.n = n
                self.p = p
