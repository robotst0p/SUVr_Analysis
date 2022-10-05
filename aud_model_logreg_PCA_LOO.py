import importlib
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import scikitplot as skplt

#normalization 
from sklearn.preprocessing import StandardScaler

#model/model_training imports
from sklearn.model_selection import train_test_split

#import LEAVEONEOUT for cross validation
from sklearn.model_selection import LeaveOneOut

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

from sklearn import metrics

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

#use PCA for feature reduction and PCA transform test sets in the loop
pca = PCA(n_components = 13)
stored_model = pca.fit(X)

y_pred_list = []
y_test_list = []

for train_index, test_index in loo.split(X):
    model = LogisticRegression(solver = 'liblinear')

    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    train_normal = scaler.fit(X_train)
    X_train_normal = train_normal.transform(X_train)
    X_test_normal = train_normal.transform(X_test)

    stored_model = pca.fit(X_train_normal)
    train_model = stored_model.transform(X_train_normal)
    test_model = stored_model.transform(X_test_normal)

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


#plot decomposition model to see how many pc's are needed to capture variance
skplt.decomposition.plot_pca_component_variance(pca)
plt.show()

#performance measures of the model in testing
print("Precision, Recall, Confusion Matrix, in testing\n")

#precision recall scores in testing
print(metrics.classification_report(y_test_list, y_pred_list, digits = 3))

#confusion matrix for testing 
print(metrics.confusion_matrix(y_test_list, y_pred_list))

#accuracy of model in testing 
print("Accuracy:", metrics.accuracy_score(y_test_list, y_pred_list))





