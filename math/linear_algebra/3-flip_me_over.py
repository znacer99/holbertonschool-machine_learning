#!/usr/bin/env python3
"""
module that returns the transpose of a 2D matrix
"""


def matrix_transpose(matrix):
    """
    Returns the transpose of a matrix
    """
    import numpy as np
    return np.transpose(matrix).tolist()
