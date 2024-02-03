#!/usr/bin/env python3

class Poisson:
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
            self.lambtha = float(sum(data) / len(data))
            if self.lambtha <= 0:
                raise ValueError("lambtha must be a positive value")

    def pmf(self, k):
        if not isinstance(k, int):
            k = int(k)
        if k < 0:
            return 0
        else:
            import math
            pmf_value = (self.lambtha ** k) * (math.exp(-self.lambtha)) / math.factorial(k)
            return pmf_value

    def cdf(self, k):
        if not isinstance(k, int):
            k = int(k)
        if k < 0:
            return 0
        else:
            import math
            cdf_value = sum([(self.lambtha ** i) * (math.exp(-self.lambtha)) / math.factorial(i) for i in range(k + 1)])
            return cdf_value
