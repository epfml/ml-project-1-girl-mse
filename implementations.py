
import numpy as np

# First we have to define the function that computes the MSE loss
def compute_mse_loss(y, tx, w):
    """
    Computes the MSE loss.

    Args:
        y:  shape=(N,)
        tx: shape=(N, D)
        w:  shape=(D,)

    Returns:
        loss: scalar
    """
    error = y - np.dot(tx, w)
    loss = 1/(2*np.shape(error)[0])*np.dot(error.T, error)[0,0]
    
    return loss

# Function that computes the gradient
def compute_gradient(y, tx, w):
    """
    Computes the gradient of MSE loss function.

    Args:
        y:  shape=(N,)
        tx: shape=(N, D)
        w:  shape=(D,)

    Returns:
        grad: vector of shape=(D,)
    """
    error = y - np.dot(tx, w)
    grad = (-1/np.shape(error)[0])*np.dot(tx.T, error)
    
    return grad   

# Linear regression using gradient descent 
def mean_squared_error_gd(y, tx, initial_w,  max_iters, gamma):
    """
    Performs the gradient descent algorithm using MSE loss to find optimal fit to data.

    Args:
        y:  shape=(N,)
        tx: shape=(N, D)
        initial_w:  shape=(D,)
        max_iters: scalar
        gamma: scalar

    Returns:
        w: shape=(D,)
        loss: scalar
    """
    #Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    
    for n_iter in range(max_iters):
        
        grad = compute_gradient(y, tx, w)
        loss = compute_mse_loss(y, tx, w)
        w = w - gamma * grad
        
        #Store w and loss
        ws.append(w)
        losses.append(loss)
    
    loss = compute_mse_loss(y, tx, w)

    return ws[-1], loss

# For stochastic gradient we only need one additional function
def compute_stoch_gradient(y, tx, w):
    """
    Computes the gradient of MSE loss function of one sample.

    Args:
        y:  shape=(N,)
        tx: shape=(N, D)
        w:  shape=(D,)

    Returns:
        grad: vector of shape=(D,)
    """
    error = y - np.dot(tx, w)
    grad = (-1)*np.dot(tx.T, error)
    
    return grad

def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """
    Performs the classical stochastic gradient descent algorithm (batch size=1)
      using MSE loss to find optimal fit to data.

    Args:
        y:  shape=(N,)
        tx: shape=(N, D)
        initial_w:  shape=(D,)
        max_iters: scalar
        gamma: scalar

    Returns:
        w: shape=(D,)
        loss: scalar
    """
    #Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        # We take one random sample for SGD
        index = np.random.randint(np.shape(tx)[0])
        x_sample = tx[index:,:]
        y_sample = y[index:,:]
        loss = compute_mse_loss(y, tx, w)
        grad = compute_stoch_gradient(y_sample, x_sample, w)
        w = w - gamma*grad
        
        ws.append(w)
        losses.append(loss)
    loss = compute_mse_loss(y, tx, w)

    return ws[-1], loss

# Least squares function for MSE loss
def least_squares(y, tx):
    """
    Finds optimal weights to fit the data using with the least squares method.

    Args:
        y:  shape=(N,)
        tx: shape=(N, D)

    Returns:
        w: shape=(D,)
        loss: scalar
    """

    w = np.linalg.solve(np.dot(tx.T,tx),np.dot(tx.T,y) )
    loss = compute_mse_loss(y, tx, w)
    
    return w, loss

# Ridge regression
def ridge_regression(y, tx, lambda_):
    """
    Performs the least squares method of finding optimal weights to fit the data 
    with MSE and ridge regression.

    Args:
        y:  shape=(N,)
        tx: shape=(N, D)
        lambda_: scalar

    Returns:
        w: shape=(D,)
        loss: scalar
    """
    N = tx.shape[0]
    I = np.eye(tx.shape[1])
    lambdaI = 2 * N * lambda_* I
    
    XtX_reg = np.dot(tx.T,tx) + lambdaI
    XtY = np.dot(tx.T,y)
    w = np.linalg.solve(XtX_reg, XtY)
    loss = compute_mse_loss(y, tx, w)

    return w, loss

# Helping functions for calculating logistic regression 
def sigmoid(t):
    """
    Applies sigmoid function on t.

    Args:
        t: scalar or numpy array

    Returns:
        scalar or numpy array

    """
    return 1 / (1 + np.exp(-t))

def calculate_logistic_loss(y, tx, w):
    """
    Computes the cost by negative log likelihood.
    Args:
        y:  shape=(N,)
        tx: shape=(N, D)
        w:  shape=(D,)

    Returns:
        a non-negative loss

    """

    return -np.mean(y*np.log(sigmoid(np.dot(tx,w))) + (1-y)*np.log(1-sigmoid(np.dot(tx,w))))
    
def calculate_logistic_gradient(y, tx, w):
    """
    Computes the gradient of logistic loss.

    Args:
        y:  shape=(N,)
        tx: shape=(N, D)
        w:  shape=(D,)

    Returns:
        a vector of shape (D,)

    """

    return tx.transpose().dot(sigmoid(tx.dot(w))-y)/y.shape[0]

def calculate_logistic_gradient_tuning(y, tx, w, w0, w1):
    """
    Computes the gradient of loss when the classes have different weights.

    Args:
        y:  shape=(N,)
        tx: shape=(N, D)
        w:  shape=(D,)
        w0: scalar, weight of class 0
        w1: scalar, weight of class 1

    Returns:
        a vector of shape (D,)

    """
    # ***************************************************
    N = len(y)
    grad = -w0*np.dot(tx.T,(1-sigmoid(np.dot(tx,w)))*y)/N + w1*np.dot(tx.T,(1-y)*sigmoid(np.dot(tx,w)))/N # 1.5, 0.85
    # ***************************************************
    return grad

def logistic_regression_step(y, tx, w, gamma):  
    """
    Returns the loss and updated weights according to gradient descent method.

    Args:
        y:  shape=(N,)
        tx: shape=(N, D)
        w:  shape=(D,)

    Returns:
        loss: scalar number
        w: updated vector of weights shape=(D,)
   
    """
    # ***************************************************
    # return loss, gradient, and Hessian:
    w = w -  gamma*calculate_logistic_gradient(y, tx, w)
    loss = calculate_logistic_loss(y, tx, w)

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
    loss = calculate_logistic_loss(y, tx, w)
    return w, loss
   
def reg_logistic_regression_step(y, tx, w, gamma, lambda_, w0, w1):
    """
    Do one step of gradient descent, using the penalized logistic regression, for different class weights.
    Return the loss and updated w.

    Args:
        y:  shape=(N,)
        tx: shape=(N, D)
        w:  shape=(D,)
        gamma: scalar
        lambda_: scalar
        w0: weight of class 0
        w1: weight of class 1

    Returns:
        loss: scalar number
        w: shape=(D,)

    """

    loss = calculate_logistic_loss(y, tx, w) + lambda_*np.linalg.norm(w)**2
    grad = calculate_logistic_gradient(y, tx, w) + 2*lambda_*w
    # ***************************************************
    # ***************************************************

    w = w - gamma*grad
    # ***************************************************
    return w, loss

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
        w, loss = reg_logistic_regression_step(y, tx, w, gamma, lambda_, 1, 1)
    # the final loss with final weights (without the penalty term)
    loss = calculate_logistic_loss(y, tx, w) 

    return w, loss

def reg_logistic_regression_tuning(y, tx, lambda_, initial_w, max_iters, gamma, w0, w1):
    """
    Does the gradient descent method of finding optimal values w, on logistic loss with class weights
    using l2 regression, starting from initial_w in max number of iterations that are given. 
    Returns final values of weights and loss (without penalty term).

    Args:
        y:  shape=(N,)
        tx: shape=(N, D)
        lambda_: scalar
        initial_w:  shape=(D,)
        max_iters: integer
        gamma: scalar
        w0: weight of class 0
        x1: weight of class 1

    Returns: 
        w: shape=(D,)
        loss: scalar 

    """
    w = initial_w
    for iter in range(max_iters):
        # get loss and update w
        w, loss = reg_logistic_regression_step(y, tx, w, gamma, lambda_, w0, w1)
    # the final loss with final weights (without the penalty term)
    loss = calculate_logistic_loss(y, tx, w) 

    return w, loss