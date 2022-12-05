#data processing libraries
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

#normalization
from sklearn.preprocessing import StandardScaler

#model/training importing 
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import RFE

#import LeaveOneOut for cross validation 
from sklearn.model_selection import LeaveOneOut

#metrics importing (accuracy, precision, sensitivity, recall)
from sklearn import metrics

#helping functions 
from helper_functions import retrieve_feature_names

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
X_normal = X_model.transform(X)

#leaveoneout cross validation and decisiontree model creation 
loo = LeaveOneOut()
loo.get_n_splits(X)

y_pred_list = []
y_test_list = []

feature_voting_list = []
rfe_feature_num = []

for i in range(0, 100):
    rfe_feature_num.append(i)


for train_index, test_index in loo.split(X):
    mod_dt = DecisionTreeClassifier(max_depth = 5, random_state = 1)

    rfe_features = 4
    rfe = RFE(estimator = mod_dt, n_features_to_select = rfe_features)

    #rfe.fit(X,y)

    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y[train_index], y[test_index]

    train_normal = scaler.fit(X_train)
    X_train_normal = pd.DataFrame(train_normal.transform(X_train), columns = X_train.columns)
    X_test_normal = pd.DataFrame(train_normal.transform(X_test), columns = X_test.columns)

    rfe.fit(X_train_normal, y_train)

    train_model = pd.DataFrame(rfe.transform(X_train_normal), columns = X_train_normal.columns[rfe.support_])
    test_model = pd.DataFrame(rfe.transform(X_test_normal), columns = X_test_normal.columns[rfe.support_])
    
    for col in train_model.columns:
        feature_voting_list.append(col)

    y_test_list.append(y_test[0])
    
    #fit the model on the training data 
    mod_dt.fit(train_model, y_train)
    
    #predict the response for the test set
    y_pred = mod_dt.predict(test_model)
    y_pred_list.append(y_pred)

#grab and display selected feature names
#feature_list = retrieve_feature_names(rfe.support_, X_df)

#print("rfe feature list: ", feature_list)

print("Accuracy:", metrics.accuracy_score(y_test_list, y_pred_list))
print("Precision:", metrics.precision_score(y_test_list, y_pred_list))
print("Recall:", metrics.recall_score(y_test_list, y_pred_list))



