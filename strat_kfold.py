#data processing libraries
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sn

#normalization
from sklearn.preprocessing import StandardScaler

#model/training importing 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import RFE

#import LeaveOneOut for cross validation 
from sklearn.model_selection import LeaveOneOut

#metrics importing (accuracy, precision, sensitivity, recall)
from sklearn import metrics

#helping functions 
from helper_functions import retrieve_feature_names, feature_vote

#parameter tuning 
import optuna

#load in suvr data as pandas dataframe
raw_dataframe = pd.read_excel('AUD_SUVr_WB.xlsx', index_col = 0)

#map subject labels to numerical values 
raw_dataframe.loc[raw_dataframe["CLASS"] == "AUD", "CLASS"] = 1
raw_dataframe.loc[raw_dataframe["CLASS"] == "CONTROL", "CLASS"] = 0

processed_data = raw_dataframe

X_df = processed_data.drop(['CLASS'], axis = 1)

#convert to numpy array for training 
#X = X_df.to_numpy()
X = X_df
y = raw_dataframe['CLASS']
y = y.astype(int)

#z-score normalization
scaler = StandardScaler()

#raw_data normalization of feature vector
X_model = scaler.fit(X)
#X_train_normal = pd.DataFrame(train_normal.transform(X_train), columns = X_train.columns)
X_normal_df = pd.DataFrame(X_model.transform(X), columns = X_df.columns)
X_normal = X_model.transform(X)

#stratified kfold cross validation
folds = 2
skf = StratifiedKFold(n_splits=folds)

score_list = []
fold_list = []

for folds in range(2, 16):
    folds = folds
    skf = StratifiedKFold(n_splits = folds)
    
    #optuna parameter tuning and data fitting
    mod_dt = DecisionTreeClassifier()
    cv = skf
    search_spaces = {"max_depth": optuna.distributions.IntDistribution(1, 100, False, 1)}
    optuna_search = optuna.integration.OptunaSearchCV(estimator = mod_dt, param_distributions = search_spaces, n_trials = 10, cv = skf, error_score = 0.0, refit = True)
    
    optuna_search.fit(X, y)
    
    y_pred = optuna_search.predict(X)
    
    score_list.append(optuna_search.best_score_)
    fold_list.append(folds)
    
plt.plot(fold_list, score_list)
plt.title('score versus num splits')
plt.xlabel('score')
plt.ylabel('accuracy')
plt.show()
    


