import numpy as np
from implementations import ridge_regression

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

def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold.

    Args:
        y:      shape=(N,)
        k_fold: K in K-fold, i.e. the fold num
        seed:   the random seed

    Returns:
        A 2D array of shape=(k_fold, N/k_fold) that indicates the data indices for each fold

    """
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
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
    weights, _ = ridge_regression(y_tr, x_tr, lambda_)

    # Calculating accuracy of model on test data
    prediction = np.array([0 if x.T @ weights < 0.5 else 1 for x in x_te])
    acc = accuracy_score(prediction, y_te)

    return weights, acc

def best_cv_ridge_reg(y, x, k_fold, lambdas, seed=1):
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
    indices = build_k_indices(y, k_fold, seed)
    best_acc = -1
    best_lambda = 0
    best_weights = []
    for ind_l, lambda_ in enumerate(lambdas):
        k_fold_accuracy = 0

        # K fold crossvalidation to compute average accuracy for that specific lambda_
        for k in range(k_fold):
            w, acc = cross_validation_ridge_reg(y, x, indices, k, lambda_)
            k_fold_accuracy += acc

        # Average accuracy
        k_fold_accuracy /= k_fold
        if(k_fold_accuracy > best_acc):
            best_acc=k_fold_accuracy
            best_lambda = lambda_

    best_weights, _ = ridge_regression(y, x, best_lambda)
    return best_lambda, best_acc, best_weights

    

