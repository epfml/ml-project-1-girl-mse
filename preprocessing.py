import numpy as np

#*********************** functions that do the data preprocessing *****************************

def build_poly_column(data, index, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree.

    Args:
        x: numpy array of shape (N,), N is the number of samples.
        degree: integer.

    Returns:
        poly: numpy array of shape (N,d+1)

    """
    # ***************************************************
    x = data[:,index]
    degrees = np.indices((len(x),degree+1))[1]
    F_poly = np.zeros((len(x),degree+1)) + x.reshape((len(x),1))
    F_poly = np.power(F_poly, degrees)
    return np.c_[data, F_poly[:,2:]]

def build_poly(data, degree):
    """polynomial basis functions for input data-data, for j=0 up to j=degree.

    Args:
        data: numpy array of shape (N,D), N is the number of samples.
        degree: integer.

    Returns:
        poly: numpy array of shape (N,d+1)

    """
    poly = data.copy()
    for i in range(data.shape[1]):
        poly = build_poly_column(poly, i, degree)
    return poly

def add_bias_column(x_train, x_test):
    """
    Adding a column of ones on the input data for bias term in weights.
    Args:
        x_train: numpy array of shape (N,D)
        x_test: numpy array of shape (N1,D)

    Returns:
        x_train_pp: numpy array of shape (N,D+1)
        x_test_pp: numpy array of shape (N1,D+1)


    """
    x_train_pp = np.c_[np.ones(x_train.shape[0]), x_train]
    x_test_pp = np.c_[np.ones(x_test.shape[0]), x_test]

    return x_train_pp, x_test_pp

def balancing_data (x_train_pp, y_train_pp, neg_label=-1, ratio_=3):
    """
    Creating a new training set with balanced data of both classes. 
    Since the class of values of 1 is smaller, it randomly samples the class of -1/0s to the length of 1s.
    Args:
        x_train: numpy array of shape (N,D)
        y_train: numpy array of shape (N,)
    Returns:
        x_train_pp: numpy array of shape (N,D+1)
        y_train_pp: numpy array of shape(N,)

    """

    N_class1 = np.sum((y_train_pp==1).astype(int))
    # Finding indexes where the output is 1
    index1 = np.argwhere(y_train_pp==1)
    index1 = np.reshape(index1, newshape=(index1.shape[0],))
    # Finding indexes where output is -1/0
    index0 = np.argwhere(y_train_pp==neg_label)
    index0 = np.reshape(index0, newshape=(index0.shape[0],))
    # Forming new input and output vectors of respective classes
    x1 = x_train_pp[index1]
    x0 = x_train_pp[index0]
    y1 = y_train_pp[index1]
    y0 = y_train_pp[index0]
    
    # Randomly choosing the data from the class of -1/0 
    ind_balanced = np.random.choice(np.arange(x0.shape[0]), N_class1*ratio_) # now we have the same amount of people who didn't have a heart attack as the ones who did
    x0_balanced = x0[ind_balanced]
    y0_balanced = y0[ind_balanced]

    # Creating the new train and test set from separate 1s and -1/0s
    x_train_pp = np.concatenate((x0_balanced,x1), axis=0)
    y_train_pp = np.concatenate((y0_balanced, y1))

    # Shuffling the data to prevent overfitting to a certain class
    ind_shuffled = np.random.permutation(np.arange(x_train_pp.shape[0]))
    x_train_pp = x_train_pp[ind_shuffled]
    y_train_pp = y_train_pp[ind_shuffled]

    return x_train_pp, y_train_pp

def column_drop(x_train, x_test):
    """
    Droping columns that have a 25% of undefined values. 
    Args:
        x_train: original training set, shape=(N,D)
        x_test: original test set, shape=(N1,D)

    Output:
        x_train_pp: training set with dropped columns, shape=(N,D*)
        x_test_pp: test set with dropped columns, shape=(N1,D*)

    """
    x_train_pp = x_train.copy()
    x_test_pp = x_test.copy()
    ratio = np.isnan(x_train_pp).sum(axis=0) / x_train_pp.shape[0]
    # Indexes we should keep
    indexes = np.where(ratio <= 0.25)[0]
    x_train_pp = x_train_pp[:, indexes]
    x_test_pp = x_test_pp[:, indexes]

    return x_train_pp, x_test_pp

def correlated_column_drop(x_train, x_test, threshold):
    """
    Droping columns that are correlated more than the certain threshold. 
    Args:
        x_train: original training set, shape=(N,D)
        x_test: original test set, shape=(N1,D)

    Output:
        x_train_pp: training set with dropped columns, shape=(N,D*)
        x_test_pp: test set with dropped columns, shape=(N1,D*)

    """
    x_train_pp = x_train.copy()
    x_test_pp = x_test.copy()
    
    correlation = np.corrcoef(x_train_pp, rowvar=False)
    correlation = np.triu(correlation, k=1)
    corr_cols = np.column_stack(np.where(np.abs(correlation)>threshold))
    
    dropped_columns = np.unique(corr_cols[:,-1])
    x_train_pp = np.delete(x_train_pp, dropped_columns, 1)
    x_test_pp = np.delete(x_test_pp, dropped_columns, 1)

    return x_train_pp, x_test_pp

def replace_w_median(x_train, x_test):
    """
    Replacing undefined values (NaN) with median value of the feature (column).
    Args:
        x_train: original training set, shape=(N,D)
        x_test: original test set, shape=(N1,D)

    Output:
        x_train_pp: shape=(N,D)
        x_test_pp: shape=(N1,D)

    """
    x_train_pp = x_train.copy()
    x_test_pp = x_test.copy()

    median_train = np.nanmedian(x_train_pp, axis=0)
    nan_indices = np.isnan(x_train_pp)
    x_train_pp[nan_indices] = median_train[np.where(nan_indices)[1]]

    median_test = np.nanmedian(x_test_pp, axis=0)
    nan_indices = np.isnan(x_test_pp)
    x_test_pp[nan_indices] = median_test[np.where(nan_indices)[1]]

    return x_train_pp, x_test_pp

def standardize(x_train, x_test):

    """
    Standardization of data with respect to mean and variance of training set.
    Args:
        x_train: input data of training set, shape=(N,D)
        x_test: input data of test set, shape=(N1,D)

    Returns:
        x_train_pp: shape=(N,D)
        x_test_pp: shape=(N1,D)
        
    """
    mean = np.mean(x_train, axis=0)
    std_dev = np.std(x_train, axis=0)
    
    x_train_pp = (x_train - mean) / (std_dev+1e-7)
    x_test_pp = (x_test - mean) / (std_dev+1e-7)

    return x_train_pp, x_test_pp

def expand_pairs(x_train, x_test, num_to_mul):
    """Expands the data with polynomials of columns up to the degree and with all pairs.

    Example:
        For columns X, Y and Z, data will be expanded with X, Y and Z up to degree and with XY, YZ, XZ.

    Returns:
        A new dataframe with newly generated combination of features.
    """
    n = x_train.shape[1]
    cur = x_train.shape[1]

    random_indices = np.random.choice(range(n), num_to_mul, replace=False)
    random_indices = np.sort(random_indices)
    
    poly = np.ones([x_train.shape[0], cur + int(num_to_mul * (num_to_mul - 1) / 2) ])
    poly_test = np.ones([x_test.shape[0], cur + int(num_to_mul * (num_to_mul - 1) / 2) ])
    
    for i in range(n):
        poly[:, i] = x_train[:, i]
        poly_test[:,i] = x_test[:,i]

    for i in random_indices:
        for j in random_indices:
            if(j>i):
                poly[:, cur] = x_train[:, i] * x_train[:, j].transpose()
                poly_test[:, cur] = x_test[:, i] * x_test[:, j].transpose()
                cur = cur + 1

    return poly, poly_test

def preprocess_data(x_train, y_train, x_test, neg_label=-1):
    """
    Function that preprocesses the training and test data. 
    The function performs the following transformations over our data:

    - dropping columns (features) that have over 50% of invalid data
    - dropping columns that are correlated more than a certain threshold
    - replacing the rest of invalid values with median
    - standardization of data
    - adding the column of zeros to produce a bias term in weights

    ********** ALTERNATEVLY: ************
    The function could also work with several exctracted valid features as taken from this reference:
    - expansion of the feature set by adding polinomial values of features
    - expansion using multiplied pairs of a random set of columns
    
    Since the test set takes values {-1,1} and our functions implement logistic regression for values {0,1} -1 are replaced with 0.

    Args:
        x_train: original training set, shape=(N,D)
        y_train: outputs of training set, shape(N,)
        x_test: test set, shape=(Ntest, D)
        neg_label: integer that represents the label of negative class, either -1 or 0

    Output:
        x_train_pp: preprocessed training set, shape=(N, D*)
        x_test_pp: preprocessed test set, shape=(Ntest, D*)
        y_train_pp: preprocessed output training vector, shape=(N,)
    """
    # relevant features: RFHYPE5, TOLDHI2, CHOLCHK, BMI5, SMOKE100, CVDSTRK3, DIABETE3, _TOTINDA, _FRTLT1, 
    # _VEGLT1, _RFDRHV5, HLTHPLN1, MEDCOST, GENHLTH, MENTHLTH, PHYSHLTH, DIFFWALK, SEX, _AGEG5YR, EDUCA, INCOME2
    #relevant_features_ind = [232, 38, 37, 253, 72, 39, 48, 284, 278, 279, 265, 30, 32, 26, 28, 27, 69, 50, 246, 257, 60]
    #relevant_features_ind = np.sort(relevant_features_ind)

    # Extraction of relevant features from input sets
    x_train_pp = x_train.copy()  #[:, relevant_features_ind] 
    x_test_pp = x_test.copy()
    y_train_pp = y_train.copy()

    # Dropping the columns with invalid values
    x_train_pp, x_test_pp = column_drop(x_train_pp, x_test_pp)

    # Dropping the correlated columns
    x_train_pp, x_test_pp = correlated_column_drop(x_train_pp, x_test_pp, 0.85)

    # Replacement of the rest of invalid values with column median
    x_train_pp, x_test_pp = replace_w_median(x_train_pp, x_test_pp)

    # Standardization of data
    x_train_pp, x_test_pp= standardize(x_train_pp, x_test_pp)

    # Replacing -1 with 0 in output if using logistic regression
    if(neg_label==0):
        y_train_pp = (y_train==1).astype(int)
    
    # Expansion of pairs
    #x_train_pp, x_test_pp = expand_pairs(x_train_pp, x_test_pp, 10)
    
    # Adding a column of ones to add the bias term to the regression
    x_train_pp, x_test_pp = add_bias_column(x_train_pp, x_test_pp)

    return x_train_pp, x_test_pp, y_train_pp

def preprocess_data_relevant(x_train, y_train, x_test, neg_label=-1):
    """
    Function that preprocesses the training and test data. 
    The function performs the following transformations over our data:

    - extracting relevant features, as taken from this reference: 
    - dropping columns that are correlated more than a certain threshold:
    - replacing the rest of invalid values with median
    - standardization of data
    - adding the column of zeros to produce a bias term in weights

    ********** ALTERNATEVLY: ************
    - expansion of the feature set by adding polinomial values of features
    - expansion using multiplied pairs of a random set of columns
    
    Since the test set takes values {-1,1} and our functions implement logistic regression for values {0,1} -1 are replaced with 0.

    Args:
        x_train: original training set, shape=(N,D)
        y_train: outputs of training set, shape(N,)
        x_test: test set, shape=(Ntest, D)
        neg_label: integer that represents the label of negative class, either -1 or 0

    Output:
        x_train_pp: preprocessed training set, shape=(N, D*)
        x_test_pp: preprocessed test set, shape=(Ntest, D*)
        y_train_pp: preprocessed output training vector, shape=(N,)
    """
    # relevant features: RFHYPE5, TOLDHI2, CHOLCHK, BMI5, SMOKE100, CVDSTRK3, DIABETE3, _TOTINDA, _FRTLT1, 
    # _VEGLT1, _RFDRHV5, HLTHPLN1, MEDCOST, GENHLTH, MENTHLTH, PHYSHLTH, DIFFWALK, SEX, _AGEG5YR, EDUCA, INCOME2
    relevant_features_ind = [232, 38, 37, 253, 72, 39, 48, 284, 278, 279, 265, 30, 32, 26, 28, 27, 69, 50, 246, 257, 60]
    relevant_features_ind = np.sort(relevant_features_ind)

    # Extraction of relevant features from input sets
    #x_train_pp = x_train.copy()  #[:, relevant_features_ind] 
    #x_test_pp = x_test.copy()
    #y_train_pp = y_train.copy()

    x_train_pp = x_train[:, relevant_features_ind] 
    x_test_pp = x_test[:, relevant_features_ind] 
    y_train_pp = y_train.copy()

    # Dropping the columns with invalid values
    #x_train_pp, x_test_pp = column_drop(x_train_pp, x_test_pp)
    x_train_pp = modify_values(x_train_pp)
    x_test_pp = modify_values(x_test_pp)

    # Dropping the correlated columns
    x_train_pp, x_test_pp = correlated_column_drop(x_train_pp, x_test_pp, 0.85)

    # Replacement of the rest of invalid values with column median
    x_train_pp, x_test_pp = replace_w_median(x_train_pp, x_test_pp)

    # Standardization of data
    x_train_pp, x_test_pp= standardize(x_train_pp, x_test_pp)

    # Replacing -1 with 0 in output if using logistic regression
    if(neg_label==0):
        y_train_pp = (y_train==1).astype(int)
    
    # Build polynomial
    x_train_pp = build_poly(x_train_pp, 3)
    x_test_pp = build_poly(x_test_pp, 3)
    # Expansion of pairs
    x_train_pp, x_test_pp = expand_pairs(x_train_pp, x_test_pp, 10)
    
    # Adding a column of ones to add the bias term to the regression
    x_train_pp, x_test_pp = add_bias_column(x_train_pp, x_test_pp)

    return x_train_pp, x_test_pp, y_train_pp

def modify_values(x_train_pp):
    # Replce with more meaningful values and remove meaningless samples
    
    # 1) _RFHYPE5

    # We switch 1(no) for 0 and 2(yes) for 1 and 3(not indicated) for 2
    x_train_pp[:,0][x_train_pp[:, 0] == 9] = 3
    x_train_pp[:, 0] = x_train_pp[:, 0] - 1
    
    # 2) TOLDHI2
    # first we delete values=7,9 because they mean don't know/no response
    # We switch 1(no) for 0 and 2(yes) for 1
    x_train_pp[:, 1][x_train_pp[:, 1] == 2] = 0
    x_train_pp[:,1][(x_train_pp[:, 1] == 9)] = 2
    x_train_pp[:,1][(x_train_pp[:,1] == 7)] = 2

    # 3) _CHOLCHK
    # first we delete values=9, because they mean don't know/no response
    # We switch 2&3 (haven't checked in past 5 years and never checked) and 1 stays the same
    x_train_pp[:, 2] = np.where(x_train_pp[:, 2] >=2, 0, 1)

    # 4) SMOKE100
    # first we remove 7&9
    x_train_pp[:,4] = np.where(x_train_pp[:,4] == 2, 0, 1)
    x_train_pp[:,4][(x_train_pp[:, 4] == 9)] = 2
    x_train_pp[:,4][(x_train_pp[:, 4] == 7)] = 2

    # 5) CVDSTRK3
    # first we remove 7&9
    x_train_pp[:,5] = np.where(x_train_pp[:,5] == 2, 0, 1)
    x_train_pp[:,5][(x_train_pp[:, 5] == 9)] = 2
    x_train_pp[:,5][(x_train_pp[:, 5] == 7)] = 2

    # 6) DIABETE3
    # 0 - no and women during pregnancy (2, 3) / 1 - yes (1) / 2 - borderline (4)
    x_train_pp[:,6][x_train_pp[:, 6] == 2] = 0
    x_train_pp[:,6][x_train_pp[:, 6] == 3] = 0
    x_train_pp[:,6][x_train_pp[:, 6] == 4] = 2
    x_train_pp[:,6][(x_train_pp[:, 6] == 9)] = 2
    x_train_pp[:,6][(x_train_pp[:, 6] == 7)] = 2

    # 7) _TOTINDA
    # remove all 9, no response
    
    # replace 2 for 0 (no physical activity)
    x_train_pp[:,7] = np.where(x_train_pp[:,7] == 2, 0, 1)
    x_train_pp[:,7][x_train_pp[:, 7] == 9] = 2
   
    # 8) _FRTLT1
    # remove all 9, no response
    # replace 2 for 0 (less than one fruit per day)
    x_train_pp[:,8] = np.where(x_train_pp[:,8] == 2, 0, 1)
    x_train_pp[:,8][x_train_pp[:, 8] == 9] = 2

    # 9) _VEGLT1
    # remove all 9, no response
    # replace 2 for 0 (less than one vegetable per day)
    x_train_pp[:,9] = np.where(x_train_pp[:,9] == 2, 0, 1)
    x_train_pp[:,9][x_train_pp[:, 9] == 9] = 2

    # 10) _RFDRHV5
    # remove all 9, no response
    x_train_pp[:,10][x_train_pp[:, 10] == 9] = 3

    # change 2 for 1 (because 2 is for heavy drinkers) and 1 for 0
    x_train_pp[:,10] = x_train_pp[:,10] - 1
    
    # 11) HLTHPLN1
    # remove all 9&7, no answer
    # 1 stays the same, change 2 to 0 (2 is having no healthcare plan)
    x_train_pp[:,11] = np.where(x_train_pp[:,11] == 2, 0, 1)
    x_train_pp[:,11][(x_train_pp[:, 11] == 9)] = 2
    x_train_pp[:,11][(x_train_pp[:,11] == 7)] = 2

    # 12) MEDCOST
    # remove 7&9, no response 
    # 1 stays the same, change 2 to 0 (2 is being able to access med. service regarding of cost)
    x_train_pp[:,12] = np.where(x_train_pp[:,12] == 2, 0, 1)
    x_train_pp[:,12][(x_train_pp[:, 12] == 9)] = 2
    x_train_pp[:,12][(x_train_pp[:,12] == 7)] = 2

    # 13) GENHLTH
    # only remove 7&9, no responses
    x_train_pp[:,13][(x_train_pp[:, 13] == 9)] = 6
    x_train_pp[:,13][(x_train_pp[:,13] == 7)] = 6

    
    # 14) MENTHLTH
    # remove 77&99, no responses
    # set 88 to 0 (it means no depressive days)
    x_train_pp[:, 14][(x_train_pp[:, 14] == 99)] = 0
    x_train_pp[:, 14][(x_train_pp[:, 14] == 77)] = 0 
    x_train_pp[:, 14][(x_train_pp[:, 14] == 88)] = 0
    
    # 15) PHYSHLTH
    # remove 77&99, no responses
    # set 88 to 0 (it means no physically bad days)
    x_train_pp[:, 15][(x_train_pp[:, 15] == 99)] = 0
    x_train_pp[:, 15][(x_train_pp[:, 15] == 77)] = 0 
    x_train_pp[:, 15][(x_train_pp[:, 15] == 88)] = 0    

    # 16) DIFFWALK
    # remove 7&9, no response
    # 1 stays the same, chnage 2 to 0 (it means no difficulties in walking)
    x_train_pp[:,16] = np.where(x_train_pp[:,16] == 2, 0, 1)
    x_train_pp[:,16][(x_train_pp[:, 16] == 9)] = 2
    x_train_pp[:,16][(x_train_pp[:,16] == 7)] = 2

    # SEX
    # we'll change because male are more prone to heart diseases than female
    x_train_pp[:,17][x_train_pp[:,17] == 2] = 0
    
    # _AGEG5YR
    # remove 14 because it means no response
    x_train_pp[:,18][(x_train_pp[:, 18] == 9)] = 14
    x_train_pp[:,18][(x_train_pp[:,18] == 7)] = 14

    # EDUCA
    # only remove 9, for no response 
    x_train_pp[:,19][x_train_pp[:,19] == 9] = 4

    
    # INCOME2
    # remove only don't know and no answer, 77 & 99
    x_train_pp[:,20][(x_train_pp[:, 20] == 9)] = 5
    x_train_pp[:,20][(x_train_pp[:,20] == 7)] = 5
   

    return x_train_pp