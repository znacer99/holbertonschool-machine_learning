#!/usr/bin/env python3
"""
Module
"""


def poly_derivative(poly):
    """
    function
    """
    if type(poly) != list or len(poly) == 0:
        return None
    if len(poly) == 1:
        return [0]
    for x in range(1, len(poly)):
        poly[x] = poly[x] * x
    poly.pop(0)
    return poly
