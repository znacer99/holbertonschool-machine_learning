#!/usr/bin/env python3
"""
module that calculates the sum of 2 arrays.
"""


def add_arrays(arr1, arr2):
    """
    calculates sum of 2 arrays
    """
    result = []
    for i in range (len(arr1)):
        for j in range (len(arr2)):
            x = arr1[i] + arr2[j]
            result.append(x)
    return result
