import numpy as np


# Write a function that takes as input a list of numbers, and returns
# the list of values given by the softmax function.
def softmax(L):
    exp_L = np.exp(L)
    return exp_L / np.sum(exp_L)


# Write a function that takes as input two lists Y, P,
# and returns the float corresponding to their cross-entropy.
def cross_entropy(Y, P):
    Y = np.array(Y)
    P = np.array(P)
    yp = Y * np.log(P)
    return - np.sum(yp + (1 - Y) * np.log(1 - P))
