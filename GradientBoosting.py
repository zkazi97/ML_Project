"""
Zain Kazi
ML Project: Random Forests
Created 11/9/20
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
from catboost import CatBoostClassifier
from catboost import CatBoostRegressor
from sklearn.ensemble import GradientBoostingClassifier
from imblearn.over_sampling import SMOTE
from sklearn import metrics
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, plot_confusion_matrix

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

#%% CATBOOST CLASSIFIER
# Define catboost object
lr = [.0001, .001, .01, .1]
auc = []
for i in lr:
    cat = CatBoostClassifier(  iterations=2000,
                               learning_rate=i,
                               depth=2    ) 
                            
    # Fit model on training set
    cat.fit(x_train, y_train)

    # Create predictions for validation and get accuracy
    cat_predictions = cat.predict(x_valid)
    cat_probs = cat.predict_proba(x_valid)
    aucscore = roc_auc_score(y_valid, cat_probs[:,1])
    print("ROC AUC Score: " + str(aucscore))
    auc.append(aucscore)

# Plot learning rate and AUC score
plt.scatter(lr, auc)
plt.xlabel("Learning Rate")
plt.ylabel("AUC")
plt.figure()

#Create dataframe of feature importance and plot 
fea_imp = pd.DataFrame({'imp': cat.feature_importances_, 'col': x_train.columns})
fea_imp = fea_imp.sort_values(['imp', 'col'], ascending=[True, False]).iloc[-30:]
fea_imp.plot(kind='barh', x='col', y='imp', figsize=(10, 7), legend=None)
plt.title('CatBoost - Feature Importance')
plt.ylabel('Features')
plt.xlabel('Importance');

# Include 10 most import features
imp_features = ['v13','v33','v1','v51','v2','v103','v127','v110','v75','v123']
x_train_features = x_train[imp_features]
x_valid_features = x_valid[imp_features]


#%% RETRAIN ON FEATURES
cat.fit(x_train_features, y_train)

# Create predictions for new validation and get accuracy
cat_predictions2 = cat.predict(x_valid_features)
cat_probs2 = cat.predict_proba(test)
print("Accuracy:", metrics.accuracy_score(y_valid,cat_predictions2))
print("ROC AUC Score: " + str(roc_auc_score(y_valid, cat_probs2[:,1])))
cat_probs2 = cat.predict_proba(x_valid)


gb_test_predictions = pd.DataFrame({'binary.target1': cat_probs2[:,1]})

gb_test_predictions.to_csv("C:/Users/zaink/Downloads/GB_Model_Probs.csv")


#%% CATBOOST REGRESSION
# Redefine target as continuous variable
y_train = train['cont.target']
y_valid = valid['cont.target']

lr = [.05, .075, .1, .125, .15, .175, .2, .225, .25, .275, .3, .325, .35, .375, .4]
mae = []

for i in lr:
# Define catboost object
    cat_reg = CatBoostRegressor(iterations=2000,
                                learning_rate=i,
                                depth=2 ) 
                            
    # Fit model on training set
    cat_reg.fit(x_train, y_train,
            eval_set=(x_valid, y_valid),
            use_best_model=True)
    
    # Create predictions for validation and get accuracy
    cat_reg_predictions = cat_reg.predict(x_valid)
    cat_errors = abs(cat_reg_predictions - y_valid)
    MAE_catReg = round(np.mean(cat_errors), 4)
    mae.append(MAE_catReg)
    print("MAE: " + str(MAE_catReg))
    
plt.scatter(lr, mae)
plt.xlabel("Learning Rate")
plt.ylabel("MAE")
plt.figure()



