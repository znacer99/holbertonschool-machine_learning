#!/usr/bin/env python3
"""
module
"""


def mat_mul(mat1, mat2):
    """
    function
    """

    if len(mat1[0]) != len(mat2):
        return None
    mat3 = [[0 for i in mat2[0]] for j in mat1]

    for i in range(len(mat1)):
        for j in range(len(mat2[0])):
            for k in range(len(mat2)):
                mat3[i][j] += mat1[i][k]*mat2[k][j]
    return mat3
