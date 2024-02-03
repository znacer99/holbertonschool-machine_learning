#!/usr/bin/env python3
import numpy as np

class Exponential:
    def __init__(self, data=None, lambtha=1.):
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            else:
                self.lambtha = float(lambtha)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = float(1 / (sum(data) / len(data)))
            if self.lambtha <= 0:
                raise ValueError("lambtha must be a positive value")

    def pdf(self, x):
        if x < 0:
            return 0
        else:
            import math
            pdf_value = self.lambtha * math.exp(-self.lambtha * x)
            return pdf_value
