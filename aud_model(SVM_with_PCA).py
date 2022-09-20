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

#use PCA for feature selection
from sklearn.decomposition import PCA
pca = PCA(n_components = 1)

X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
explained_variance = pca.explained_variance_ratio_

print(explained_variance)
#import and train model on aud uptake data
from sklearn import svm

#create classifier 
clf = svm.SVC(kernel = 'linear')

#train the model
clf.fit(X_train, y_train)

#predict the response for test set
y_pred = clf.predict(X_test)


#performance measures of the model

from sklearn import metrics
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

print("Precision:", metrics.precision_score(y_test, y_pred))

print("Recall:", metrics.recall_score(y_test, y_pred))