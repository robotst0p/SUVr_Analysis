#data processing libraries
import pandas as pd
import numpy as np 

#model/training importing 
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.decomposition import PCA

#import LeaveOneOut for cross validation 
from sklearn.model_selection import LeaveOneOut

#metrics importing (accuracy, precision, sensitivity, recall)
from sklearn import metrics

#load in suvr data as pandas dataframe
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

#leaveoneout cross validation and decisiontree model creation 
loo = LeaveOneOut()
loo.get_n_splits(X)

mod_dt = DecisionTreeClassifier(max_depth = 25, random_state = 1)

for i in range(1, 27):
    #use PCA for feature reduction and PCA transform test sets in the loop
    pca = PCA(n_components = i)
    stored_model = pca.fit(X)
    
    y_pred_list = []
    y_test_list = []

    print("PCA Component Number:\n", i)

    for train_index, test_index in loo.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        train_model = stored_model.transform(X_train)
        test_model = stored_model.transform(X_test)

        y_test_list.append(y_test[0])

        #fit the model on the training data 
        mod_dt.fit(train_model, y_train)
        
        #predict the response for the test set

        y_pred = mod_dt.predict(test_model)
        y_pred_list.append(y_pred)

    print("Accuracy:", metrics.accuracy_score(y_test_list, y_pred_list))
    print("Precision:", metrics.precision_score(y_test_list, y_pred_list, zero_division = 1))
    print("Recall:", metrics.recall_score(y_test_list, y_pred_list, zero_division = 1))


