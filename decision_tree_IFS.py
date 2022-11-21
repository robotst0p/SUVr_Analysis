#data processing libraries
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sn

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
from helper_functions import retrieve_feature_names, feature_vote

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
final_feature_list = []

for train_index, test_index in loo.split(X):
    mod_dt = DecisionTreeClassifier(max_depth = 5, random_state = 1)
    
    rfe_features = 4
    rfe = RFE(estimator = mod_dt, n_features_to_select = rfe_features)
    
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

#input voting selection threshold as percentage value (percentage of times feature needs to be selected by rfe)
threshold = .4        
final_feature_list, voting_dict = feature_vote(feature_voting_list, rfe_features, threshold, X_df)
    
for train_index, test_index in loo.split(X):
    mod_dt = DecisionTreeClassifier(max_depth = 5, random_state = 1)

    rfe_features = 4
    rfe = RFE(estimator = mod_dt, n_features_to_select = rfe_features)

    #rfe.fit(X,y)

    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y[train_index], y[test_index]

    #normalization of training and testing X
    train_normal = scaler.fit(X_train)
    X_train_normal = pd.DataFrame(train_normal.transform(X_train), columns = X_train.columns)
    X_test_normal = pd.DataFrame(train_normal.transform(X_test), columns = X_test.columns)
    
    #use voted on features to transform training and testing X
    X_train_normal = X_train_normal.loc[:, final_feature_list]
    X_test_normal = X_test_normal.loc[:, final_feature_list]
    
    #fitting of recursive feature eliminator to normalized X and y training sets
    #rfe.fit(X_train_normal, y_train)

    #train_model = pd.DataFrame(rfe.transform(X_train_normal), columns = X_train_normal.columns[rfe.support_])
    #test_model = pd.DataFrame(rfe.transform(X_test_normal), columns = X_test_normal.columns[rfe.support_])

    y_test_list.append(y_test[0])
    
    #fit the model on the training data 
    mod_dt.fit(X_train_normal, y_train)
    
    #predict the response for the test set
    y_pred = mod_dt.predict(X_test_normal)
    y_pred_list.append(y_pred)

#grab and display selected feature names
#feature_list = retrieve_feature_names(rfe.support_, X_df)

#print("rfe feature list: ", feature_list)

corr_matrix = X_train_normal.corr()
sn.heatmap(corr_matrix, annot=True)
plt.show()

plt.bar(voting_dict.keys(), voting_dict.values(), color ='blue',width = .5)



print("Accuracy:", metrics.accuracy_score(y_test_list, y_pred_list))
print("Precision:", metrics.precision_score(y_test_list, y_pred_list))
print("Recall:", metrics.recall_score(y_test_list, y_pred_list))



