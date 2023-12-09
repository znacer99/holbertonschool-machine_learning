#!/usr/bin/env python3
"""
Module
"""


def np_shape(matrix):
    """
    function
    """
    shape_tuple = ( )
    current_dim = matrix

    while type(current_dim) == list:
        shape_tuple += (len(current_dim),)
        current_dim = current_dim[0]

    return shape_tuple
