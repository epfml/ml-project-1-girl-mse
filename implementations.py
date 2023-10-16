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