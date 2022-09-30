import pandas as pd
import numpy as np
import matplotlib as plt

#model/training imports 
from sklearn.model_selection import train_test_split
from sklearn import svm

#metrics importing(percision, accuracy, sensitivity, recall)
from sklearn import metrics

#load in suvr data as pandas dataframe
raw_dataframe = pd.read_excel('AUD_SUVr_WB.xlsx', index_col = 0)

#map subject labels to numerical values 
raw_dataframe.loc[raw_dataframe["CLASS"] == "AUD", "CLASS"] = 0
raw_dataframe.loc[raw_dataframe["CLASS"] == "CONTROL", "CLASS"] = 1

processed_data = raw_dataframe

X = processed_data.drop(['CLASS'], axis = 1)

#convert to numpy array for training
X = X.to_numpy()
y = raw_dataframe['CLASS']
y = y.astype(int)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.5, random_state = 42)

#create classifier 
clf = svm.SVC(kernel = 'linear')

#train the model
clf.fit(X_train, y_train)

#predict the response for test set
y_pred = clf.predict(X_test)

#performance measures of the model
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

print("Precision:", metrics.precision_score(y_test, y_pred, zero_division = 1))

print("Recall:", metrics.recall_score(y_test, y_pred, zero_division = 1))


