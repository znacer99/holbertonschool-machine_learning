#!/usr/bin/env python3
"""
module to calculate size of a matrix
"""

def matrix_shape(matrix):
    """
    Matrix shape function : this function calculates the length
    of the first dimension of the Matrix and then retrieves the first element
    to calculate the next dimension.
    """

    size = []
    if isinstance(matrix, list):
        size.append(len(matrix))

        element = matrix[0]
        if isinstance(element, list):
            size.extend(matrix_shape_2(element, size))
    return size


def matrix_shape_2(matrix, size):
    """
    Matrix_shape_2 is the matrix shape function that works recursively
    with the first matrix shape function and it's the same function but
    it's recursively.
    """

    if isinstance(matrix, list):
        size.append(len(matrix))

        element = matrix[0]
        if isinstance(element, list):
            size.extend(matrix_shape_2(element, size))
    return size
