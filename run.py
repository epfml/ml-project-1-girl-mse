import numpy as np
import matplotlib.pyplot as plt
from helpers import *
from preprocessing import preprocess_data, preprocess_data_relevant
from implementations import *
from utils import accuracy_score, f1_score, cross_validation_ridge_reg, best_cv_ridge_reg, cross_validation_log_reg, best_cv_log_reg ,build_k_indices

# Importing the data
x_train, x_test, y_train, train_ids, test_ids = load_csv_data("resources/dataset_to_release")

# Optimal values of parameters
lambdas = [0.1]
gammas = [0.1]
w0s = [1]
w1s = [1]

# Data preprocessing and setup for accuracy prediction using cross validation
x_train_pp, x_test_pp, y_train_pp = preprocess_data(x_train, y_train, x_test, neg_label=0)
initial_w = np.zeros((x_train_pp.shape[1],))
k_fold = 5
max_iters = 25

for w0 in w0s:
    for w1 in w1s:
       _, _, best_acc, best_weights, best_f1, _, _ = best_cv_log_reg(y_train_pp, x_train_pp, k_fold, lambdas, gammas, max_iters, [w0], [w1], balancing=True)
       print(f'For the w0: {w0}, and w1: {w1} we get F1 score of: {best_f1}, and accuracy of: {best_acc}')

# Predicting with the values of optimal weights 

y_prediction = np.array([-1 if sigmoid(x.T @ best_weights) < 0.5 else 1 for x in x_test_pp])

# How many values of each class the method is predicting 

print('There are this much -1s: ', np.sum((y_prediction==-1).astype(int)))
print('There are this much 1s: ', np.sum((y_prediction==1).astype(int)))

# Submission
create_csv_submission(test_ids, y_prediction, "final_prediction.csv")