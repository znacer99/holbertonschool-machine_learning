#!/usr/bin/env python3
"""
Module
"""


def np_slice(matrix, axes={}):
    """
    main
    """
    alist = []
    for i in range(len(matrix.shape)):
        if i in axes:
            alist.append(slice(*axes[i]))
        else:
            alist.append(slice(None))
    atuple = tuple(alist)
    return matrix[atuple]
