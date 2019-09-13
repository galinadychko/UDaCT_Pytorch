#!/usr/bin/env python
# coding: utf-8
#  ipython modules/perceptron_trick.py


import numpy as np
import os
import sys
from typing import Dict, Tuple, Union, List, Iterable

from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import rc

from IPython import get_ipython
from argparse import ArgumentParser


WORKING_DIR = os.getcwd()
SAVE_PATH = WORKING_DIR + "/data/"
SAVE_NAME = "data.csv"


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# Step function, which helps us to convert each prediction score to the coresspining class labels.
def stepFunction(x: Union[np.ndarray, float]) -> Union[np.ndarray, int]:
    """
    Convert negative values into 0 and not negative into 1.

    Parameters
    ----------
    :param x: np.ndarray OR float: input object for conversion
    
    Returns
    --------
    np.ndarray
              Array, where 0 located at places with negative value and -1 - otherwise.
    """
    if (type(x) == int) or (type(x) == float):
        return 1 - int(x < 0)
    else:
        return 1 - (x < 0).astype(int)


def prediction(X: np.ndarray, 
               W: np.ndarray, 
               b: float) -> np.ndarray:
    """
    Define the most suitable class based on features of each observation
    
    Parameters
    ----------
    :param X: np.ndarray: Data matrix, where each row corresponds to the 
                          appropriate object and each column - to its feature
    :param W: np.ndarray: Weights of linear separator in format: 
                            np.array([A, B]),
                          while model is described by the following way
                            Ax + By + b = 0
    :param b: float: Bias of linear separator
    
    Returns
    ---------
    np.ndarray
              Predicted class label(s) for the input object(s)
    """
    pred = X.dot(W.T) + b
    return stepFunction(pred).reshape(-1)


def perceptronStep(X: np.ndarray,
                   y: np.ndarray,
                   W: np.ndarray,
                   b: float,
                   learn_rate: float) -> Tuple[np.ndarray, float]:
    """
    Update input model`s parameters so that,
    the classification result accurasy increases

    Parameters
    ----------
    :param X: np.ndarray: Data matrix, where each row corresponds to the
                          appropriate object and each column - to its feature
    :param y: np.ndarray: array of class labels, oreder of which corresponds
                          to the X matrix`s order.
                          Values should be 1 or 0.
    :param W: np.ndarray: Weights of linear separator in format:
                            np.array([A, B]),
                          while model is described in the following way
                            Ax + By + b = 0
    :param b: float OR int: Bias of linear separator
    :learn_rate: float OR int: the  extent of weights updating

    Returns
    ---------
    (np.ndarray, float)
                       Updated weights and bias of the input linear model
    """
    n_obs, n_feat = 1, 1
    if not isinstance(W, np.ndarray):
        raise TypeError("Not correct W type")
    if len(X.shape) == 2:
        n_obs, n_feat = X.shape
    if len(X.shape) == 1:
        n_obs = len(X)
    if n_obs != len(y):
        raise ValueError("Different number of observations in X and y data")

    W = np.float64(W)
    b = float(b)

    y_pred = prediction(X, W, b)
    W_minus_ind = np.where((y == 0) & (y_pred != y))[0]
    W_minus = - learn_rate * np.sum(X[W_minus_ind], axis=0)

    W_plus_ind = np.where((y == 1) & (y_pred != y))[0]
    W_plus = learn_rate * np.sum(X[W_plus_ind], axis=0)

    b = b + learn_rate * (len(W_plus_ind) - len(W_minus_ind))

    return W + W_minus + W_plus, b

# This function runs the perceptron algorithm repeatedly on the dataset,<br/>
# and returns a few of the boundary lines obtained in the iterations,<br/>
# for plotting purposes.<br/>
# ***
# Feel free to play with the learning rate and the num_epochs,
# and see your results plotted below.<br/>

# In[5]:


def trainPerceptronAlgorithm(X: np.ndarray, 
                             y: np.ndarray, 
                             learn_rate: float = 0.01,
                             num_epochs: int = 25) -> List:
    """
    Find the most suitable parameters of linear model for class separation task
    
    Parameters 
    ----------
    :param X: np.ndarray: Data matrix, where each row corresponds to the 
                          appropriate object and each column - to its feature
    :param y: np.ndarray: array of class labels, which corresponds to the X matrix`s observations.
                          Values should be 1 or 0.
    :param learn_rate: float: the  extent of weights updating
    :param num_epochs: int: number of steps for searching the most suitable parameters
    
    Returns
    -------
    List
        The most suitable model parameters for each epoch
    """
    if not isinstance(X, np.ndarray):
        raise TypeError("Not correct X type")
    if not isinstance(y, np.ndarray):
        raise TypeError("Not correct y type")
    if not isinstance(learn_rate, (float, int)):
        raise TypeError("Not correct learn_rate type")
    if not isinstance(num_epochs, int):
        raise TypeError("Not correct num_epochs type")
        
    x_min, x_max = min(X.T[0]), max(X.T[0])
    np.random.seed(42)
    W = np.random.rand(1, 2)
    np.random.seed(42)
    b = np.random.rand(1)[0] + x_max
    # These are the solution lines that get plotted below.
    boundary_lines = []
    for i in range(num_epochs):
        # In each epoch, we apply the perceptron step.
        W, b = perceptronStep(X, y, W, b, learn_rate)
        boundary_lines.append((-W[0, 0]/W[0, 1], -b/W[0, 1]))
    return boundary_lines


# plot data objects and boundary line
def plot_separation(X_data: np.ndarray,
                    y_data: np.ndarray,
                    line_params: Dict[str, float]) -> None:
    rc('axes', edgecolor='lightgray')
    fig = plt.figure(figsize=(8, 6))

    ax = fig.add_subplot()

    ax.scatter(X_data[:, 0], X_data[:, 1], c=y_data.astype(int))

    line_ax = np.arange(0., 1., .1)
    line_ay = line_ax * line_params["k"] + line_params["b"]

    ax.plot(line_ax, line_ay, ls="solid", c="red")

    miss_clasified_id = np.where(y_data != prediction(X_data, np.array([line_params["k"], -1]), line_params["b"]))
    ax.scatter(X_data[miss_clasified_id, 0], X_data[miss_clasified_id, 1], c="red", alpha=0.1, s=145)

    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")

    legend_elements = [Line2D([0], [0], color='r', lw=4,
                              label='Boundary line\ny={}x + {}'.format(round(line_params["k"], 2), round(line_params["b"]))),
                       Line2D([0], [0], color="w",
                              marker="o", markerfacecolor="yellow", label='Class 1'),
                       Line2D([0], [0], color="w",
                              marker="o", markerfacecolor="#5e0087", label='Class 0'),
                       Line2D([0], [0], color="w",
                              marker="o", markerfacecolor="#FF9933", markeredgecolor="#FFCCCC",
                              markersize=10, markeredgewidth=3, label='Missclassified')
                       ]
    plt.grid(alpha=0.3)
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1., 1.))
    plt.title("Classification model\nbased on the perceptron trick", fontdict={"size": 20})
    plt.show()


def main(argv):
    parser = ArgumentParser()
    parser.add_argument("--save-path", "-sp", action="store", default=SAVE_PATH,
                        help="Path for saving file with data")
    parser.add_argument("--save-name", "-sn", action="store", default=SAVE_NAME,
                        help="Name for saving file with data")
    args = parser.parse_args(argv)

    os.system("python {}/tools/load_data.py --save-path={} --save-name={}".format(WORKING_DIR, args.save_path, args.save_name))
    # load data
    data = np.genfromtxt(args.save_path + args.save_name, delimiter=',')

    # split it into features and labels
    X = data[:, :2]
    y = data[:, 2]

    # take the model of the last epoch
    boundary_lines = trainPerceptronAlgorithm(X, y, learn_rate=0.01, num_epochs=100)
    final_line = boundary_lines[-1]

    # make computation for line plotting
    # plot the results
    plot_separation(X_data=X, y_data=y,
                    line_params={"k": final_line[0], "b": final_line[1]})


if __name__ == "__main__":
    main(sys.argv[1:])
