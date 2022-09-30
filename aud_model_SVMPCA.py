import importlib
import pandas as pd
import numpy as np
import matplotlib as plt

#TRAIN SET WILL BE OF SHAPE (100X27) (100 features 27 samples)

#model/model training imports 
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn import svm

#metrics importing (accuracy, percision, sensitivity, recall)
from sklearn import metrics

#reshape data .reshape(1,-1)
#aaa = pca.fit(X)
#qqq = aaa.transofrm(X)
#qq2 = aaa.transform(X[test_index].reshape(-1,1))
#z score normalization 

#load in suvr data as pandas dataframe
raw_dataframe = pd.read_excel('AUD_SUVr_WB.xlsx', index_col = 0)

#map subject labels to numerical values a
raw_dataframe.loc[raw_dataframe["CLASS"] == "AUD", "CLASS"] = 1
raw_dataframe.loc[raw_dataframe["CLASS"] == "CONTROL", "CLASS"] = 0

processed_data = raw_dataframe

X = processed_data.drop(['CLASS'], axis = 1)

#convert to numpy array for training
X = X.to_numpy()
y = raw_dataframe['CLASS']
y = y.astype(int)
    

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.5, random_state = 42)

#use PCA for feature selection
pca = PCA(n_components = 1)

X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
explained_variance = pca.explained_variance_ratio_

print("Explained Variance Scores:")
print(explained_variance)

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