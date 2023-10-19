# Implementations of functions from lab session 5

import numpy as np

# Helping functions for calculating logistic regression 
def sigmoid(t):
    """apply sigmoid function on t.

    Args:
        t: scalar or numpy array

    Returns:
        scalar or numpy array

    """
    return np.divide(np.exp(t),(1+np.exp(t)))

def calculate_loss(y, tx, w):
    """compute the cost by negative log likelihood.
    Implementation changed from the ones in labs to fit the vector implementation requirements of the project.
    Args:
        y:  shape=(N,)
        tx: shape=(N, D)
        w:  shape=(D,)

    Returns:
        a non-negative loss

    """
    assert y.shape[0] == tx.shape[0]
    assert tx.shape[1] == w.shape[0]

    # ***************************************************
    # INSERT YOUR CODE HERE
    # TODO
    N = len(y)
    loss = -np.dot(np.transpose(y), np.dot(tx,w)) + np.sum(np.log(np.ones(N,) + np.exp(np.dot(tx,w))))
    loss /= N
    return loss

def calculate_gradient(y, tx, w):
    """compute the gradient of loss.

    Args:
        y:  shape=(N,)
        tx: shape=(N, D)
        w:  shape=(D,)

    Returns:
        a vector of shape (D, 1)

    """
    # ***************************************************
    # INSERT YOUR CODE HERE
    # TODO
    N = len(y)
    grad = np.dot(np.transpose(tx), sigmoid(np.dot(tx,w))-y)/N
    # ***************************************************
    return grad

   

def logistic_regression_step(y, tx, w, gamma):
    """returns the loss and updated weights according to gradient descent method.

    Args:
        y:  shape=(N,)
        tx: shape=(N, D)
        w:  shape=(D,)

    Returns:
        loss: scalar number
        w: updated vector of weights shape=(D,)
   
    """
    # ***************************************************
    # INSERT YOUR CODE HERE
    # return loss, gradient, and Hessian: TODO
    loss = calculate_loss(y, tx, w)
    w = w -  gamma*calculate_gradient(y, tx, w)
    # ***************************************************
    return loss, w

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """
    Does the gradient descent method of finding optimal values w, starting from initial_w in 
    max number of iterations that are given. Returns final values of weights and loss.

    Args:
        y:  shape=(N,)
        tx: shape=(N, D)
        initial_w:  shape=(D,)
        max_iters: integer
        gamma: scalar

    Returns: 
        w: shape=(D,)
        loss: scalar number

    """
    w = initial_w
    for iter in range(max_iters):
        # get loss and update w
        loss, w = logistic_regression_step(y, tx, w, gamma)
    loss = calculate_loss(y, tx, w)
    return w, loss

    
def reg_logistic_regression_step(y, tx, w, gamma, lambda_):
    """
    Do one step of gradient descent, using the penalized logistic regression.
    Return the loss and updated w.

    Args:
        y:  shape=(N,)
        tx: shape=(N, D)
        w:  shape=(D,)
        gamma: scalar
        lambda_: scalar

    Returns:
        loss: scalar number
        w: shape=(D,)

    """
    # ***************************************************
    # INSERT YOUR CODE HERE
    # return loss, gradient: TODO
    loss = calculate_loss(y, tx, w) + lambda_*np.linalg.norm(w)**2
    grad = calculate_gradient(y, tx, w) + 2*lambda_*w
    # ***************************************************
    # ***************************************************
    # INSERT YOUR CODE HERE
    # update w: TODO
    w = w - gamma*grad
    # ***************************************************
    return loss, w

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """
    Does the gradient descent method of finding optimal values w, using l2 regression,
    starting from initial_w in max number of iterations that are given. 
    Returns final values of weights and loss (without penalty term).

    Args:
        y:  shape=(N,)
        tx: shape=(N, D)
        lambda_: scalar
        initial_w:  shape=(D,)
        max_iters: integer
        gamma: scalar

    Returns: 
        w: shape=(D,)
        loss: scalar 

    """
    w = initial_w
    for iter in range(max_iters):
        # get loss and update w
        loss, w = reg_logistic_regression_step(y, tx, w, gamma, lambda_)
    # the final loss with final weights (without the penalty term)
    loss = calculate_loss(y, tx, w) 

    return w, loss