#!/usr/bin/env python3
"""
Defines Normal class that represents normal distribution
"""

class Normal:
    """
    Class that represents normal distribution
    """

    def __init__(self, data=None, mean=0., stddev=1.):
        """
        Class constructor

        parameters:
            data [list]: data to be used to estimate the distribution
            mean [float]: the mean of the distribution
            stddev [float] : the standard deviation of the distribution
        """
        if data is None:
            if stddev < 1:
                raise ValueError("stddev must be a positive value")
            else:
                self.stddev = float(stddev)
                self.mean = float(mean)
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            elif len(data) < 2:
                raise ValueError("data must contain multiple values")
            else:
                mean = float(sum(data) / len(data))
                self.mean = mean
                summation = 0
                for x in data:
                    summation += ((x - mean) ** 2)
                stddev = (summation / len(data)) ** (1 / 2)
                self.stddev = stddev
