import pandas as pd
import numpy as np
import matplotlib as plt

#load in suvr data as pandas dataframe
raw_dataframe = pd.read_excel('AUD_SUVr_WB.xlsx', index_col = 0)

#map subject labels to numerical values 

raw_dataframe.loc[raw_dataframe["CLASS"] == "AUD", "CLASS"] = 0
raw_dataframe.loc[raw_dataframe["CLASS"] == "CONTROL", "CLASS"] = 1

processed_data = raw_dataframe

from sklearn.model_selection import train_test_split

X = processed_data.drop(['CLASS'], axis = 1)

#convert to numpy array for training

X = X.to_numpy()
y = raw_dataframe['CLASS']
y = y.astype(int)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.5, random_state = 42)

#import and train model on aud uptake data

from sklearn.linear_model import LogisticRegression

print("TEST DATA")
print(y)

log_reg = LogisticRegression()
log_reg.fit(X,y)

#training predictions
training_prediction = log_reg.predict(X_train)

#test predictions
test_prediction = log_reg.predict(X_test)

#performance measures of the model

from sklearn import metrics
print("Precision, Recall, Confusion matrix, in training\n")

#precision recall scores
print(metrics.classification_report(y_train, training_prediction, digits = 3))

#confusion matrix 
print(metrics.confusion_matrix(y_train, training_prediction))

#performance in testing 
print("Precision, Recall, Confusion matrix, in testing\n")

#precision recall scores
print(metrics.classification_report(y_test, test_prediction, digits=3))

#confusion matrix for testing 
print(metrics.confusion_matrix(y_test, test_prediction))

print("Accuracy:", metrics.accuracy_score(y_test, test_prediction))


