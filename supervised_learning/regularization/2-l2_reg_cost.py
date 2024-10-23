#!/usr/bin/env python3
"""
calculates the cost of a neural network with L2 regularization
"""


import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


def l2_reg_cost(cost, lam):
    """
    cost of nn using L2 regularization
    """
     # Get all trainable variables (weights)
    weights = tf.trainable_variables()

    # Compute L2 regularization term: sum of squared weights multiplied by lambda
    l2_loss = tf.add_n([tf.nn.l2_loss(w) for w in weights])

    # Add L2 regularization to the original cost
    total_cost = cost + lam * l2_loss

    return total_cost
