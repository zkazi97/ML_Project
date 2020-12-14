"""
Zain Kazi
ML Project: Ridge and LASSO Regression
Created 11/11/20
"""

#%% IMPORT PACKAGES
# Data manipulation
import pandas as pd
import numpy as np
import tensorflow as tf
import math

# Data visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Modeling and plotting
from imblearn.over_sampling import SMOTE
from sklearn import metrics
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, plot_confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier, MLPRegressor
from keras.layers import Dense,Activation
from keras.models import Sequential

#%% IMPORT DATA
train = pd.read_csv("C:/Users/zaink/documents/Python Scripts/MLProject21_train.csv")
valid = pd.read_csv("C:/Users/zaink/documents/Python Scripts/MLProject21_valid.csv")
test = pd.read_csv("C:/Users/zaink/documents/Python Scripts/MLProject21_test.csv")

#%% DATA CLEANING
# Create Column Index
columns = train.columns 

# Convert categorical to dummy variables
train = pd.get_dummies(train)
# head = train.head(100)

# Separate predictors from targets
x_train = train.drop(columns = ['binary.target1', 'binary.target2', 'cont.target'])
y_train = train['binary.target1']

# Oversample to get 50/50 target split
pre_proportion = y_train.value_counts()/y_train.count() 
# sm = SMOTE(random_state = 12, sampling_strategy = 1)
# x_train, y_train = sm.fit_sample(x_train,y_train)
# post_proportion = y_train.value_counts()/y_train.count()

# Repeat for validation set
valid = pd.get_dummies(valid)
x_valid = valid.drop(columns = ['binary.target1', 'binary.target2', 'cont.target'])
y_valid = valid['binary.target1']

# Standardize data for neural networks
scaler = StandardScaler()
scaler.fit(x_train)
x_train_scaled = scaler.transform(x_train)
x_valid_scaled = scaler.transform(x_valid)

test = pd.get_dummies(test)
x_test_scaled = scaler.transform(test)

#%% NEURAL NETWORK ClASSIFIER
# Define neural networks
nnc = MLPClassifier(solver = 'sgd',
                   alpha = .2,
                   hidden_layer_sizes = (10,10),
                   activation = 'logistic',
                   max_iter = 2000)

# Fit model 
nnc.fit(x_train_scaled, y_train)

# Predict on validation and evaluate
nnc_predictions = nnc.predict(x_valid_scaled)
nnc_probs = nnc.predict_proba(x_valid_scaled)
print("ROC AUC Score: " + str(roc_auc_score(y_valid, nnc_probs[:,1])))

nnc_test_predictions = nnc.predict_proba(x_test_scaled)



#%% NEURAL NETWORK REGRESSOR
# Change target variable to continous variable 
y_train = train['cont.target']
y_valid = valid['cont.target']

# Define neural networks
nnr = MLPRegressor(solver = 'sgd',
                   alpha = .2,
                   hidden_layer_sizes = (10,10,10),
                   activation = 'relu',
                   max_iter = 2000)

# Fit model 
nnr.fit(x_train_scaled, y_train)

# Predict on validation and evaluate
nnr_predictions2 = nnr.predict(x_valid_scaled)
nnr_errors = abs(nnr_predictions2 - y_valid)
mae_nnr = round(np.mean(nnr_errors), 4)
print("MAE: " + str(mae_nnr))

nnr_test_predictions = pd.DataFrame({'cont.target': nnr.predict(x_test_scaled)})


nnr_test_predictions.to_csv("C:/Users/zaink/Downloads/NN_Regr_Model.csv")
