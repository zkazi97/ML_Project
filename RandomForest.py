"""
Zain Kazi
ML Project: Random Forests
Created 11/8/20
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
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score, classification_report, confusion_matrix, plot_confusion_matrix

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

# Summary Stats
summary = train.describe(include = 'all').transpose()
corr = train.corr()
proportion = y_valid.value_counts()/y_valid.count()

#%% RF CLASSIFIER
# Define Random Forest Instance
mtry = round(math.sqrt(len(x_train.columns)))
rf = RandomForestClassifier(n_estimators = 10, max_features = mtry)

# Train model using predictors and target
rf.fit(x_train,y_train)

# Create predictions array and print accuracy using validation
rf_predictions = rf.predict(x_valid)
rf_probs = rf.predict_proba(x_valid)
print(classification_report(y_valid, rf_predictions))
print("Accuracy:", metrics.accuracy_score(y_valid,rf_predictions))
# print("ROC AUC Score: " + str(roc_auc_score(y_valid, rf_probs[:,1])))


# Find optimal probability cutoff
fpr, tpr, thresholds = metrics.roc_curve(y_valid, rf_probs[:, -1])
false_pos_rate, true_pos_rate, proba = roc_curve(y_valid, rf_probs[:, -1])
plt.figure()
plt.plot([0,1], [0,1], linestyle="--") # plot random curve
plt.plot(false_pos_rate, true_pos_rate, marker=".", label=f"AUC = {roc_auc_score(y_valid, rf_probs[:, -1])}")
plt.title("ROC Curve")
plt.ylabel("True Positive Rate")
plt.xlabel("False Positive Rate")
plt.legend(loc="lower right")

# Find max accuracy based on probability cutoff
acc = []
for i in range(1,100):
    decisions = (rf_probs >= i/100).astype(int)
    a = pd.Series(decisions[:,1])
    acc.append(metrics.accuracy_score(y_valid,a))
print("Max Accuracy is " + str(max(acc)))


#%% RETRAIN ON SELECTED FEATURES
featureRank = pd.Series(rf.feature_importances_, index = x_train.columns).sort_values(ascending=False)
rf_fea_imp = pd.DataFrame({'imp': rf.feature_importances_, 'col': x_train.columns})
rf_fea_imp = rf_fea_imp.sort_values(['imp', 'col'], ascending=[True, False]).iloc[-30:]
rf_fea_imp.plot(kind='barh', x='col', y='imp', figsize=(10, 7), legend=None)
plt.title('RF - Feature Importance')
plt.ylabel('Features')
plt.xlabel('Importance');

# Retrain model and obtain accuracy based on most important features
imp_feats = featureRank[featureRank > .01].index.tolist()
x_train_features = x_train[imp_feats]
x_valid_features = x_valid[imp_feats]
mtry = len(x_train_features.columns)
rf = RandomForestClassifier(n_estimators = 100, max_features = mtry)
rf.fit(x_train_features,y_train)
rf_predictions2 = rf.predict(x_valid_features)
print("Accuracy:", metrics.accuracy_score(y_valid,rf_predictions2))
print(classification_report(y_valid, rf_predictions2))


#%% RF REGRESSION
# Redefine target as continuous variable
y_train_features = train['cont.target']
y_valid_features = valid['cont.target']

# Define Random Forest Instance
mtry_reg = round(math.sqrt(len(x_train.columns)))
rfReg = RandomForestRegressor(n_estimators = 100, max_features = mtry_reg)

# Train model using predictors and target
rfReg.fit(x_train,y_train)

# Create predictions array and print accuracy using validation
rfReg_predictions = rfReg.predict(x_valid)
rf_errors = abs(rfReg_predictions - y_valid)
MAE_rrReg = round(np.mean(rf_errors), 4)

