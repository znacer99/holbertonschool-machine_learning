#!/usr/bin/env python3
"""
module that calculates the sum of 2 arrays.
"""


def add_arrays(arr1, arr2):
    """
    calculates sum of 2 arrays
    """
    result = []
    if not arr1 and not arr2:
        return result
    if len(arr1) != len(arr2):
        return result
    for i in range (len(arr1)):
        x = arr1[i] + arr2[i]
        result.append(x)
    return result
