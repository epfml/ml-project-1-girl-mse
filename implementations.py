import numpy as np

# First we have to define the function that computes the MSE loss
def compute_mse_loss(y, tx, w):
    error = y - np.dot(tx, w)
    loss = 1/(2*np.shape(error)[0])*np.dot(error.T, error)
    
    return loss

# Function that computes the gradient
def compute_gradient(y, tx, w):
    error = y - np.dot(tx, w)
    grad = (-1/np.shape(error)[0])*np.dot(tx.T, error)
    
    return grad   

# Linear regression using gradient descent 
def mean_squared_error_gd(y, tx, initial_w,  max_iters, gamma):
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
        
    return losses[-1], ws[-1] # NOTE: Razlikuje se u tome sto ovo nase treba da vrati samo poslednje vrednosti

# For stochastic gradient we only need one additional function
def compute_stoch_gradient(y, tx, w):
    error = y - np.dot(tx, w)
    grad = (-1)*np.dot(tx.T, error)
    
    return grad

# Linear regression using stochastic gradient descent - batch_size = 1
def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
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
    
    return losses[-1], ws[-1] # NOTE: Razlikuje se u tome sto ovo nase treba da vrati samo poslednje vrednosti

# Least Squares
def compute_loss(y, tx, w):
    N = len(y)
    txt = tx.T
    txt_w = tx @ w
    e2 = (y - txt_w)@(y - txt_w)
    loss = 1/N * np.sum(e2)
    return loss

def least_squares(y, tx):
    least_square = np.linalg.solve(np.dot(tx.T,tx),np.dot(tx.T,y) )
    mse = compute_loss(y, tx, least_square)
    
    return least_square, mse

def least_squares(y, tx):
    least_square = np.linalg.solve(np.dot(tx.T,tx),np.dot(tx.T,y) )
    mse = compute_loss(y, tx, least_square)
    
    return least_square, mse

# Ridge regression
def ridge_regression(y, tx, lambda_):
    N = tx.shape[0]
    I = np.identity(tx.shape[1])
    lambdaI = 2 * N * lambda_* I
    
    XtX_reg = tx.T @ tx + lambdaI
    XtY = tx.T @ y
    w = np.linalg.solve(XtX_reg, XtY)
    loss = compute_loss(y, tx, w)
    return loss, w

# Logistic regression
def sigmoid_function(t):
    return 1.0/(1 + np.exp(-t))

def neg_log_likelihood(y, tx, w):
    pred = sigmoid_function(tx @ w)
    mat1 = np.dot(y.T ,np.log(pred))
    mat2 = np.dot((1 - y).T, (np.log(1 - pred)))
    loss = mat1 + mat2
    res = np.squeeze(-loss).item() 
    return (1 / y.shape[0]) * res


def logistic_gradient(y, tx, w):
    
    pred = sigmoid_function(tx @ w)
    grad = tx.T @ (pred - y)
    return (1 / y.shape[0]) * grad

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    
    w = initial_w
    losses=[]
    for iter in range(max_iters):
        loss = neg_log_likelihood(y, tx, w)
        losses.append(loss)
        grad = logistic_gradient(y, tx, w)
        w -= gamma * grad
        
    return w, losses


# Logistic regression with L2 regularization term

def sigmoid_function(t):
    return 1.0 / (1 + np.exp(-t))

def neg_log_likelihood(y, tx, w, lambda_):
    pred = sigmoid_function(tx @ w)
    mat1 = np.dot(y.T, np.log(pred))
    mat2 = np.dot((1 - y).T, np.log(1 - pred))
    loss = mat1 + mat2
    regularization = 0.5 * lambda_ * np.sum(w**2)
    res = np.squeeze(-loss + regularization).item()
    
    return (1 / y.shape[0]) * res

def reg_logistic_gradient(y, tx, w, lambda_):
    pred = sigmoid_function(tx @ w)
    grad = tx.T @ (pred - y)
    regularization = lambda_ * w
    return (1 / y.shape[0]) * (grad + regularization)

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    w = initial_w
    losses = []
    for iter in range(max_iters):
        loss = neg_log_likelihood(y, tx, w, lambda_)
        losses.append(loss)
        grad = reg_logistic_gradient(y, tx, w, lambda_)
        w -= gamma * grad
    return w, losses