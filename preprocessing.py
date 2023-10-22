import numpy as np

#*********************** functions that do the data preprocessing *****************************

def preprocess_data(x_train, y_train, x_test, y_test=[]):
    """
    Function that preprocesses the training and test data. 
    If the test data represents validation data, use y_test as the outputs of valudation dataset. Otherwise it is empty.
    Out of 80 features in the initial training and test set, only the relevant ones to the heart disease are extracted.
    Since the test set takes values {-1,1} and our functions implement logistic regression for values {0,1} -1 are replaced with 0.

    Args:
        x_train: original training set, shape=(N,D)
        y_train: outputs of training set, shape(N,)
        x_test: test set, shape=(Ntest, D)

    Output:
    """
    # relevant features: RFHYPE5, TOLDHI2, CHOLCHK, BMI5, SMOKE100, CVDSTRK3, DIABETE3, _TOTINDA, _FRTLT1, 
    # _VEGLT1, _RFDRHV5, HLTHPLN1, MEDCOST, GENHLTH, MENTHLTH, PHYSHLTH, DIFFWALK, SEX, _AGEG5YR, EDUCA, INCOME2
    relevant_features_ind = [232, 38, 37, 253, 72, 39, 48, 284, 278, 279, 265, 30, 32, 26, 28, 27, 69, 50, 246, 257, 60]
    # Extraction of relevant features from input sets
    x_train_pp = x_train[:, relevant_features_ind]
    x_test_pp = x_test[:, relevant_features_ind]

    # Replacing -1 with 0 in output test
    y_train_pp = (y_train==1).astype(int)
    if (y_test != []):
        y_test_pp = (y_test==1).astype(int)
    else:
        y_test_pp = []
    # Removing rows with NaN values
    y_train_pp = y_train_pp[~np.isnan(x_train_pp).any(axis=1)]
    x_train_pp = x_train_pp[~np.isnan(x_train_pp).any(axis=1)]

    return x_train_pp, x_test_pp, y_train_pp, y_test_pp
    