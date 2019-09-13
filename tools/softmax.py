import numpy as np


# Write a function that takes as input a list of numbers, and returns
# the list of values given by the softmax function.
def softmax(L):
    exp_L = np.exp(L)
    return exp_L / np.sum(exp_L)
