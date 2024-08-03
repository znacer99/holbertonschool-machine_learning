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

    def factorial(self, lst):
        """
        finds factorial
        """
        fact, facts = 1, []
        for k in lst:
            for x in range(1, k + 1):
                fact = fact * x
            facts.append(int(fact))
            fact = 1
        return tuple(facts)

    def pmf(self, k):
        """
        Calculates the value of the PMF for a given number of "successes"
        """
        if type(k) is not int:
            k = int(k)
        if k < 0 or k > self.n:
            return 0
        n, x, n_x = self.factorial([self.n, k, self.n-k])
        choose = n / (x * n_x)
        p = (self.p**k) * ((1-self.p)**(self.n-k))
        return choose * p

    def cdf(self, k):
        """
        Calculates the value of the CDF for a given number of successes
        """
        if type(k) is not int:
            k = int(k)
        if k < 0 or k > self.n:
            return 0
        return sum([self.pmf(i) for i in range(k + 1)])
