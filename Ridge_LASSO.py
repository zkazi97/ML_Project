"""
Zain Kazi
ML Project: Ridge and LASSO Regression
Created 11/11/20
"""

#%% IMPORT PACKAGES
# Data manipulation
import pandas as pd
import numpy as np
import math

# Data visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Modeling and plotting
from imblearn.over_sampling import SMOTE
from sklearn import metrics
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression

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
y_train = train['binary.target2']

# Oversample to get 50/50 target split
pre_proportion = y_train.value_counts()/y_train.count() 
# sm = SMOTE(random_state = 12, sampling_strategy = 1)
# x_train, y_train = sm.fit_sample(x_train,y_train)
# post_proportion = y_train.value_counts()/y_train.count()

# Repeat for validation set
valid = pd.get_dummies(valid)
x_valid = valid.drop(columns = ['binary.target1', 'binary.target2', 'cont.target'])
y_valid = valid['binary.target2']

#%% RIDGE REGRESSION MODEL
# Define lists of lambda values and respective MAEs
MAE = []
lambda_vals = [2.95]

# Model ridge regression for each lambda value and store MAE
for i in lambda_vals:
    rr = Ridge(alpha = i, normalize = True)
    rr.fit(x_train, y_train)
    rr_predictions = rr.predict(x_valid)
    errors = abs(rr_predictions - y_valid)
    MAE.append(round(np.mean(errors), 5))
vals = pd.DataFrame(MAE, lambda_vals)
minLambda = round(vals[0].idxmin(),3)
minMae = min(MAE)
print('The minimum MAE is ' + str(minMae) + ' with a lambda of ' + str(minLambda))


# Plot lambda value and mean absolute error 
sns.set_style("darkgrid")
sns.scatterplot(x = lambda_vals, y = MAE)
plt.ylim(.2,.3)
plt.title('MAE vs. Lambda')
plt.xlabel('Lambda')
plt.ylabel('Mean Absolute Error')

#%% LASSO REGRESSION MODEL
# Define lists of lambda values and respective MAEs
MAE = []
lambda_vals = [.016]

# Model ridge regression for each lambda value and store MAE
for i in lambda_vals:
    l = Lasso(alpha = i, normalize = True)
    l.fit(x_train, y_train)
    l_predictions = l.predict(x_valid)
    errors = abs(l_predictions - y_valid)
    MAE.append(round(np.mean(errors), 6))
vals = pd.DataFrame(MAE, lambda_vals)
minLambda = round(vals[0].idxmin(),3)
minMae = min(MAE)
print('The minimum MAE is ' + str(minMae) + ' with a lambda of ' + str(minLambda))


# Plot lambda value and mean absolute error 
sns.set_style("darkgrid")
sns.scatterplot(x = lambda_vals, y = MAE)
plt.ylim(.2,.21)
plt.title('MAE vs. Lambda')
plt.xlabel('Lambda')
plt.ylabel('Mean Absolute Error')


