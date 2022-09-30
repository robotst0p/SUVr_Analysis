import importlib
from re import X 
import pandas as pd
import numpy as np 
import matplotlib as plt

#model/model_training imports
from sklearn.model_selection import train_test_split

#import LEAVEONEOUT for cross validation
from sklearn.model_selection import LeaveOneOut

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

from sklearn import metrics

raw_dataframe = pd.read_excel('AUD_SUVr_WB.xlsx', index_col = 0)

#map subject labels to numerical values 
raw_dataframe.loc[raw_dataframe["CLASS"] == "AUD", "CLASS"] = 1
raw_dataframe.loc[raw_dataframe["CLASS"] == "CONTROL", "CLASS"] = 0

processed_data = raw_dataframe

X = processed_data.drop(['CLASS'], axis = 1)

#convert to numpy array for training 
X = X.to_numpy()
y = raw_dataframe['CLASS']
y = y.astype(int)

#leaveoneout cross validation and lin reg model creation 
loo = LeaveOneOut()
loo.get_n_splits(X)

model = LogisticRegression(solver = 'liblinear')

for i in range(1, 27):
    #use PCA for feature reduction and PCA transform test sets in the loop
    pca = PCA(n_components = i)
    stored_model = pca.fit(X)

    y_pred_list = []
    y_test_list = []

    print("PCA Component Num:\n", i)

    for train_index, test_index in loo.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        train_model = stored_model.transform(X_train)
        test_model = stored_model.transform(X_test)

        #fit the model on the training data
        model.fit(train_model, y_train)

        #predict the response for the test set
        y_pred = model.predict(test_model)

        #fit the model on the training data
        model.fit(train_model, y_train)
        
        #training predictions
        training_prediction = model.predict(train_model)

        #test predictions
        test_prediction = model.predict(test_model)

        y_test_list.append(y_test[0])
        y_pred_list.append(test_prediction[0])

    #performance measures of the model in testing
    print("Precision, Recall, Confusion Matrix, in testing\n")

    #precision recall scores in testing
    print(metrics.classification_report(y_test_list, y_pred_list, digits = 3))

    #confusion matrix for testing 
    print(metrics.confusion_matrix(y_test_list, y_pred_list))

    #accuracy of model in testing 
    print("Accuracy:", metrics.accuracy_score(y_test_list, y_pred_list))





