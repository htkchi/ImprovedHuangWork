#!/usr/bin/env python

import random
import numpy as np
import scipy.sparse as sp

import mip
from mip import Model, CutType, OptimizationStatus


def cosine_similarity(x, y):
    assert x.shape == y.shape and len(x.shape) == 1, \
        AssertionError(f"Check the shape of input vector with x = {x.shape}, y= {y.shape}")
    # return np.abs(x @ y) / (np.linalg.norm(x) * np.linalg.norm(y))
    return x @ y / (np.linalg.norm(x) * np.linalg.norm(y))


def cosine_similarity_from_dict(x_dict, y_dict, x_keys):
    x = np.array([x_dict.get(x_key, 0) for x_key in x_keys])
    y = np.array([y_dict.get(x_key, 0) for x_key in x_keys])
    assert x.shape == y.shape, AssertionError(f"Check the shape of input vector with x = {x.shape}, y= {y.shape}")
    # return np.abs(x @ y) / (np.linalg.norm(x) * np.linalg.norm(y))
    return x @ y / (np.linalg.norm(x) * np.linalg.norm(y))


def euclidean_distance(x, w, b):
    return np.abs(w @ x + b) / np.linalg.norm(w)


def create_model(name, A, b, c, integral_array, sense):
    """Create python-mip solution model.

    :param name: name of the mip model.
    :param type: str
    
    :param A, b, c: 
        arrays for a standard constraint programming, in form of `min/max c @ x, s.t. A @ x <= b`.
    :param type: np.array
        A should be of shape (num_constraints, num_vars)
        b should be of shape (num_constraints,)
        c should be of shape (num_vars,)

    :param integral_array: 
        array for whether the variable is integral. `integral_array[i] > 0` means the i-th variable
        in the problem is integral, otherwise continuous.
    :param type:
        np.array of shape (num_vars,)

    :param sense: sense of the objective, "MIN" or "MAX".
    :param type: str

    :returns: (m, x, constr)
        m: mip.Model object of this problem
        x: the variable list of this problem
        constr: the constraints list of this problem
    :rtype:
        m: mip.Model
        x: list of mip.Var
        constr: list of mip.Constraint
    """
    # Params
    num_constrs, num_vars = A.shape

    # Create a new model
    m = Model(sense=sense, solver_name=mip.CBC)

    # Create variables
    x = []
    for i, x_type in enumerate(integral_array.tolist()):
        x_type = mip.INTEGER if x_type > 0 else mip.CONTINUOUS
        x.append(m.add_var(var_type=x_type, name=f"x_{i}"))

    # Set objective
    m.objective = mip.xsum(c[i] * x[i] for i in range(num_vars))

    # Add constraints
    constr = [m.add_constr(mip.xsum(A[i, j] * x[j] for j in range(num_vars)) <= b[i],
                        name=f"C_{i}") for i in range(num_constrs)]

    return m, x, constr
