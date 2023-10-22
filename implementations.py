
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
    loss = 1/(2*np.shape(error)[0])*np.dot(error.T, error)
    
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
        x_sample = tx[index, :]
        #print(x_sample)
        y_sample = y[index]
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
    
    XtX_reg = tx.T @ tx + lambdaI
    XtY = tx.T @ y
    w = np.linalg.solve(XtX_reg, XtY)
    loss = compute_mse_loss(y, tx, w)

    return w, loss

# Helping functions for calculating logistic regression 
def sigmoid(t):
    """apply sigmoid function on t.

    Args:
        t: scalar or numpy array

    Returns:
        scalar or numpy array

    """
    return np.divide(np.exp(t),(1+np.exp(t)))

def calculate_logistic_loss(y, tx, w):
    """compute the cost by negative log likelihood.
    Implementation changed from the ones in labs to fit the vector implementation requirements of the project.
    Args:
        y:  shape=(N,)
        tx: shape=(N, D)
        w:  shape=(D,)

    Returns:
        a non-negative loss

    """
    # ***************************************************
    # INSERT YOUR CODE HERE
    # TODO
    N = len(y)
    loss = -np.dot(np.transpose(y), np.dot(tx,w)) + np.sum(np.log(np.ones(N,) + np.exp(np.dot(tx,w))))
    loss /= N
    return loss

def calculate_logistic_gradient(y, tx, w):
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
    loss = calculate_logistic_loss(y, tx, w)
    w = w -  gamma*calculate_logistic_gradient(y, tx, w)
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
    loss = calculate_logistic_loss(y, tx, w) + lambda_*np.linalg.norm(w)**2
    grad = calculate_logistic_gradient(y, tx, w) + 2*lambda_*w
    # ***************************************************
    # ***************************************************
    # INSERT YOUR CODE HERE
    # update w: TODO
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
        w, loss = reg_logistic_regression_step(y, tx, w, gamma, lambda_)
    # the final loss with final weights (without the penalty term)
    loss = calculate_logistic_loss(y, tx, w) 

    return w, loss