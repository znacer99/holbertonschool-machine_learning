#!/usr/bin/env python3
"""
module that returns the transpose of a 2D matrix
"""


def matrix_transpose(matrix):
    """
    Returns the transpose of a matrix
    """
    transpose = []
    for j in range(len(matrix[0])):
        x = []
        for i in range(len(matrix)):
            x.append(matrix[i][j])
        transpose.append(x)
    return transpose
