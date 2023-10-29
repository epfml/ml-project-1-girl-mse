[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/U9FTc9i_)

# CSE-433 Machine Learning Project 1 
## MICHD Classification 
### Results: 
In the final result of this project we obtained a 0.847 classification accuracy and an F1 score of 0.413. 
### Team:
The Girl MSE Team is composed of the following members:

Sara Zatezalo: @sarazatezalo

Marija Zelic: @masazelic

Elena Mrdja: @elena-mrdja

### Guidelines:

The project report MLProject.pdf contains the full description of the project. The code for our project has been divided into preprocessing.py: containing the functions for data preprocessing), implementations.py: containing the functions for the 6 machine leaning models, and utils.py: containing the additional functions we used in model training and validation

In order to reproduce the results of the project, place your files for training and testing should into a folder called dataset_to_release which is placed inside the folder called resources. Alternatively, change the input filepath on the line 9 in the run.py to the location of the data.

### The Project:
The aim of this project is to implement basic Machine learning models such as Linear, Ridge, and Logistic regression, in order to use them to predict whether a person is at a high risk for deeloping a Myocardial Infarct or Coronary Heart Disease (MICHD). Since the BRFSS dataset which was used in this project contains many missing values as well as features with non-informative entries, the success of the Machine Learning models heavily relied on data preprocessing. The final result of our project was obtained using Regularized Logistic Regression on the preprocessed data, which resulted in a validation accuracy of 0.847 and an F1 score of 0.413.

### Folder structure
```
├── implementaions.py: Implementations of 6 ML function
├── run.py: Python file for regenerating our final prediction.
├── preprocessing.py: Preprocessing functions including dropping missing values, replacing with median, dropping the correlated columns, adding a bias column etc.
├── utils.py: All of the utils functions such as MSE and cross-entropy, gradient calculation function, K-cross validation etc.
├── helper.py: Functions to help load the data.
├── MLProject.pdf: a 2-pages long report of the project.
├── README.md
└── resources
    ├── dataset_to_release
```

