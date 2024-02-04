#!/usr/bin/env python3
import numpy as np
from scipy.special import erf

class Normal:
    def __init__(self, data=None, mean=0., stddev=1.):
        if data is None:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            else:
                self.mean = float(mean)
                self.stddev = float(stddev)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.mean = sum(data) / len(data)
            self.stddev = (sum([(x - self.mean) ** 2 for x in data]) / len(data)) ** 0.5
            if self.stddev <= 0:
                raise ValueError("stddev must be a positive value")

    def z_score(self, x):
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        return self.mean + z * self.stddev

    def pdf(self, x):
        """Calculates the value of the PDF for a given x-value"""
        aux = ((x - self.mean) / self.stddev)**2
        result = (1 / (self.stddev * (2 * pi)**(1/2))) * e**((-1/2) * aux)
        return result

    def cdf(self, x):
        """Calculates the value of the CDF for a given x-value"""
        aux = (x - self.mean)/(self.stddev * (2**(1/2)))
        result = (1/2) * (1 + erf(aux))
        return result
