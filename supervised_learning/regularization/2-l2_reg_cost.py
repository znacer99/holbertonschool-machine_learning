#!/usr/bin/env python3
"""
calculates the cost of a neural network with L2 regularization
"""


import tensorflow as tf


def l2_reg_cost(cost, lam):
    """
    cost of nn using L2 regularization
    """
    l2_reg_cost = tf.losses.get_regularization_losses()
    return (cost + l2_reg_cost)
