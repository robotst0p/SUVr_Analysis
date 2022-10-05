import importlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scikitplot as skplt

#normalization
from sklearn.preprocessing import StandardScaler

#model/model training imports 
from sklearn.model_selection import train_test_split

#import LeaveOneOut for cross validation
from sklearn.model_selection import LeaveOneOut

from sklearn.decomposition import PCA
from sklearn import svm

#metrics importing (accuracy, percision, sensitivity, recall)
from sklearn import metrics

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

#z-score normalization 
scaler = StandardScaler()

#leaveoneout cross validation and SVM model creation
loo = LeaveOneOut()
loo.get_n_splits(X)

#use PCA for feature reduction and PCA transform test sets in the loop
pca = PCA(n_components = 9)

y_pred_list = []
y_test_list = []

for train_index, test_index in loo.split(X):
    clf = svm.SVC(kernel = 'rbf')
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    train_normal = scaler.fit(X_train)
    X_train_normal = train_normal.transform(X_train)
    X_test_normal = train_normal.transform(X_test)

    stored_model = pca.fit(X_train)
    train_model = stored_model.transform(X_train)
    test_model = stored_model.transform(X_test)

    y_test_list.append(y_test[0])

    #fit the model on the training data
    clf.fit(train_model, y_train)

    #predict the response for the test set 
    y_pred = clf.predict(test_model)
    y_pred_list.append(y_pred)

skplt.decomposition.plot_pca_component_variance(pca)
plt.show()

print("Accuracy:", metrics.accuracy_score(y_test_list, y_pred_list))
print("Precision:", metrics.precision_score(y_test_list, y_pred_list, zero_division = 1))
print("Recall:", metrics.recall_score(y_test_list, y_pred_list, zero_division = 1))

