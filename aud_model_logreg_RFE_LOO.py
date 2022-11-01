import importlib
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

#normalization 
from sklearn.preprocessing import StandardScaler

#model/model_training imports
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

#import LeaveOneOut for cross validation
from sklearn.model_selection import LeaveOneOut

#metrics importing (accuracy, precision, recall)
from sklearn import metrics

#helping functions 
from helper_functions import retrieve_feature_names

#load in suvr data as pandas dataframe
raw_dataframe = pd.read_excel('AUD_SUVr_WB.xlsx', index_col = 0)

#map subject labels to numerical values 
raw_dataframe.loc[raw_dataframe["CLASS"] == "AUD", "CLASS"] = 1
raw_dataframe.loc[raw_dataframe["CLASS"] == "CONTROL", "CLASS"] = 0

processed_data = raw_dataframe

X = processed_data.drop(['CLASS'], axis = 1)

#z_score normalization 
scaler = StandardScaler()

#convert to numpy array for training 
X = X.to_numpy()
y = raw_dataframe['CLASS']
y = y.astype(int)

#leaveoneout cross validation and lin reg model creation 
loo = LeaveOneOut()
loo.get_n_splits(X)

y_pred_list = []
y_test_list = []

for train_index, test_index in loo.split(X):
    model = LogisticRegression(solver = 'liblinear')

    rfe_features = 2
    rfe = RFE(estimator = model, n_features_to_select = rfe_features)

    rfe.fit(X,y)

    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    train_normal = scaler.fit(X_train)
    X_train_normal = train_normal.transform(X_train)
    X_test_normal = train_normal.transform(X_test)

    train_model = rfe.transform(X_train_normal)
    test_model = rfe.transform(X_test_normal)

    #fit the model on the training data
    model.fit(train_model, y_train)

    #predict the response for the test set
    y_pred = model.predict(test_model)
    
    #training predictions
    training_prediction = model.predict(train_model)

    #test predictions
    test_prediction = model.predict(test_model)

    y_test_list.append(y_test[0])
    y_pred_list.append(test_prediction[0])

#grab and display selected feature names
feature_list = retrieve_feature_names(rfe.support_, X_df)

print("rfe feature list: ", feature_list)

#performance measures of the model in testing
print("Precision, Recall, Confusion Matrix, in testing\n")

#precision recall scores in testing
print(metrics.classification_report(y_test_list, y_pred_list, digits = 3))

#confusion matrix for testing 
print(metrics.confusion_matrix(y_test_list, y_pred_list))

#accuracy of model in testing 
print("Accuracy:", metrics.accuracy_score(y_test_list, y_pred_list))





