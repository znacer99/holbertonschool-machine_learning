#!/usr/bin/env python3
"""
module to calculate size of a matrix
"""

def matrix_shape(matrix):
    """
    calculates size of a matrix
    """
    shape = []
    while isinstance(matrix, list):
        shape.append(len(matrix))
        matrix = matrix[0]
    return shape
