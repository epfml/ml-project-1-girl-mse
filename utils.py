import numpy as np
from implementations import *
from preprocessing import balancing_data

def accuracy_score(y_prediction, y_truth):
    """
    Calculates the accuracy score for given set of true values and predictions.

    Args:
        y_prediction: shape=(N,)
        y_truth: shape=(N,)

    Returns:
        accuracy = scalar
    """
    return np.sum(y_prediction==y_truth) / y_prediction.shape[0]

def f1_score(y_prediction, y_truth, label=1):
    """
    Calculates F1 score for given set of true values and predictions.

    Args:
        y_prediction: shape=(N,)
        y_truth: shape=(N,)

    Returns:
        f1_score = scalar
    """
    tp = np.sum((y_prediction==label) & (y_truth==label))
    fp = np.sum((y_truth!=label) & (y_prediction==label))
    fn = np.sum((y_prediction!=label) & (y_truth==label))

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * (precision * recall)/(precision+recall)

    return f1

def build_k_indices(y, k_fold):
    """build k indices for k-fold.

    Args:
        y:      shape=(N,)
        k_fold: K in K-fold, i.e. the fold num

    Returns:
        A 2D array of shape=(k_fold, N/k_fold) that indicates the data indices for each fold

    """
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(100)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval : (k + 1) * interval] for k in range(k_fold)]

    return np.array(k_indices)

def cross_validation_ridge_reg(y, x, k_indices, k, lambda_):
    """
    Function that performs a single cross validation on kth fold and outputs the weights obtained by least squares 
    with ridge reguralization method and accuracy for that fold.
    Args:
        y: shape=(N,)
        x: shape(N,D)
        k_indices: 2D indices for training and test set
        k: scalar, the index of fold used for testing
        lambda_: parameter of ridge regression

    Returns:
        w: best model parameters fitted to the given data
        acc: accuracy of the current fold model

    """
    # Exctracting indexes of train and test with respect to the kth fold
    train_ind = np.delete(k_indices, k, 0)
    train_ind = np.resize(train_ind, new_shape=(np.shape(train_ind)[0]*np.shape(train_ind)[1],))
    test_ind = k_indices[k]

    # Forming the training and testing sets
    x_tr = x[train_ind]
    x_te = x[test_ind]
    y_tr = y[train_ind]
    y_te = y[test_ind]

    # Fitting the model of logistic regression to the data
    x_tr, y_tr = balancing_data(x_tr, y_tr, -1)
    weights, _ = ridge_regression(y_tr, x_tr, lambda_)

    # Calculating accuracy of model on test data
    prediction = np.array([-1 if x.T @ weights < 0 else 1 for x in x_te])
    acc = accuracy_score(prediction, y_te)
    f1 = f1_score(prediction,y_te)

    return weights, acc, f1

def best_cv_ridge_reg(y, x, k_fold, lambdas):
    """
    Cross validation to estimate accuracy of model for different values of lambda parameter in ridge regression.

    Args:
        y: shape=(N,)
        x: shape(N,D)
        k_fold: number of folds, integer
        lambdas: array, parameter of ridge regression

    Returns:
        best_lambda: best model parameter lambda
        best_acc: accuracy of the best model

    """
    indices = build_k_indices(y, k_fold)
    best_acc = -1
    best_f1 = -1
    best_lambda = 0
    best_weights = []
    for ind_l, lambda_ in enumerate(lambdas):
        k_fold_accuracy = 0
        f1_score_k = 0

        # K fold crossvalidation to compute average accuracy for that specific lambda_
        for k in range(k_fold):
            w, acc, f1 = cross_validation_ridge_reg(y, x, indices, k, lambda_)
            k_fold_accuracy += acc
            f1_score_k += f1

        # Average accuracy
        k_fold_accuracy /= k_fold
        f1_score_k /= k_fold

        if(f1_score_k > best_f1):
            best_acc= k_fold_accuracy
            best_f1= f1_score_k
            best_lambda = lambda_
    
    x, y = balancing_data(x, y, -1)
    best_weights, _ = ridge_regression(y, x, best_lambda)
    return best_lambda, best_acc, best_weights, best_f1

# Cross validation for logistic and penalized logistic regression
def cross_validation_log_reg(y, x, k_indices, k, gamma, lambda_, max_iters, w0, w1, balancing=False):
    """
    Function that performs a single cross validation on kth fold and outputs the weights obtained by logistic regression
    Args:
        y: shape=(N,)
        x: shape(N,D)
        k_indices: 2D indices for training and test set
        k: scalar, the index of fold used for testing
        lambda_: parameter of ridge regression
        max_iters: integer, maximal number of iterations for gradient descent
        w0: weight for class 0
        w1: weight for class 1

    Returns:
        w: best model parameters fitted to the given data
        acc: accuracy of the current fold model

    """
    # Exctracting indexes of train and test with respect to the kth fold
    train_ind = np.delete(k_indices, k, 0)
    train_ind = np.resize(train_ind, new_shape=(np.shape(train_ind)[0]*np.shape(train_ind)[1],))
    test_ind = k_indices[k]

    # Forming the training and testing sets
    x_tr = x[train_ind]
    x_te = x[test_ind]
    y_tr = y[train_ind]
    y_te = y[test_ind]

    initial_w = np.zeros((x_tr.shape[1],))
    # Fitting the model of logistic regression to the data
    if(balancing):
        x_tr, y_tr = balancing_data(x_tr, y_tr, 0)
    w, loss = reg_logistic_regression_tuning(y_tr, x_tr, lambda_, initial_w, max_iters, gamma, w0, w1)

    # Calculating accuracy of model on test data
    prediction = np.array([0 if sigmoid(x.T @ w) < 0.5 else 1 for x in x_te])
    acc = accuracy_score(prediction, y_te)
    f1 = f1_score(prediction,y_te)

    return w, acc, f1

def best_cv_log_reg(y, x, k_fold, lambdas, gammas, max_iters, w0, w1, balancing=False):
    """
    Cross validation to estimate accuracy of model for different values of parameters for regularized logistic regression.
    While training, we balance the data by adding the samples of smaller class again to the data and shuffle it.

    Args:
        y: shape=(N,)
        x: shape(N,D)
        k_fold: number of folds, integer
        lambdas: array, parameter of ridge regression
        gammas: array, parameter of ridge regression
        max_iters: int, maximal number of iterations for gradient descent
        w0: weight for class 0
        w1: weight for class 1

    Returns:
        best_lambda: best model parameter lambda
        best_acc: accuracy of the best model

    """
    indices = build_k_indices(y, k_fold)
    best_acc = -1
    best_f1 = -1
    best_gamma = 0
    best_lambda = 0
    best_weights = []
    best_w0 = 0
    best_w1 = 0

    for i in range(len(w0)):
        for ind_l, lambda_ in enumerate(lambdas):
            for ind_g, gamma in enumerate(gammas):

                k_fold_accuracy = 0
                f1_score_k = 0

                # K fold crossvalidation to compute average accuracy for that specific lambda_
                for k in range(k_fold):
                    w, acc, f1 = cross_validation_log_reg(y, x, indices, k, gamma, lambda_, max_iters, w0[i], w1[i], balancing)
                    k_fold_accuracy += acc
                    f1_score_k += f1

                # Average accuracy
                k_fold_accuracy /= k_fold
                f1_score_k /= k_fold

                if(f1_score_k > best_f1):
                    best_acc= k_fold_accuracy
                    best_f1= f1_score_k
                    best_gamma = gamma
                    best_lambda = lambda_
                    best_w0 = w0[i]
                    best_w1 = w1[i]

    initial_w = np.zeros((x.shape[1],))
    if(balancing):
        x, y = balancing_data(x, y, 0)
    best_weights, _ = reg_logistic_regression_tuning(y, x, best_lambda, initial_w, max_iters, best_gamma, best_w0, best_w1)

    return best_lambda, best_gamma, best_acc, best_weights, best_f1, best_w0, best_w1

