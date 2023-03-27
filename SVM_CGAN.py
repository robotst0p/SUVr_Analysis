#data processing libraries
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sn
import random

#normalization
from sklearn.preprocessing import StandardScaler

#model/training importing 
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.feature_selection import RFE

#import LeaveOneOut for cross validation 
from sklearn.model_selection import LeaveOneOut

#metrics importing (accuracy, precision, sensitivity, recall)
from sklearn import metrics

#parameter tuning 
import optuna

#load in suvr data as pandas dataframe
raw_dataframe = pd.read_excel('AUD_SUVr_wb_cingulate.xlsx', index_col = 0)
synth_raw_frame = pd.read_excel('generated_Cingulate_SUVR.xlsx')

#map subject labels to numerical values 
raw_dataframe.loc[raw_dataframe["CLASS"] == "AUD", "CLASS"] = 1
raw_dataframe.loc[raw_dataframe["CLASS"] == "CONTROL", "CLASS"] = 0

processed_data = raw_dataframe

X_df = processed_data.drop(['CLASS'], axis = 1)
synth_frame_x = synth_raw_frame.drop(['Class'], axis = 1)
synth_frame_y = synth_raw_frame['Class']

synth_frame_y = synth_frame_y.astype(int)

#convert to numpy array for training 
X = X_df.copy()
y = raw_dataframe['CLASS']
y = y.astype(int)

synth_y_container = pd.Series()
synth_x_container = pd.DataFrame(columns = X.columns)
#z-score normalization
scaler = StandardScaler()

#raw_data normalization of feature vector
X_model = scaler.fit(X)
X_normal = pd.DataFrame(X_model.transform(X), columns = X_df.columns)

#leaveoneout cross validation and decisiontree model creation 
loo = LeaveOneOut()
loo.get_n_splits(X)

y_pred_list = []
y_test_list = []

#store accuracies to determine which synthetic data points to add to dataset
gan_acc_list = []

#setting initial f1 score to compare changes to 
threshold_f1 = 0
#track f1 change due to synthetic sample being added to training set if f1 is higher than threshold
f1_change = 0

y_test_list = []
filtered_test_list = []
y_pred_list = []

synth_dict = {}



for row in list(synth_frame_x.index.values):
    for train_index, test_index in loo.split(X):
        svc = svm.SVC(kernel = "rbf", random_state = 42)    
        
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        y_test_list.append(y_test)
        
        #append synthetic point to X_training set
        synth_train_x = synth_frame_x.loc[row]
        X_train = X_train.append(synth_train_x, ignore_index = True)
        
        #append synthetic point to y_training set
        synth_train_y = synth_frame_y.loc[row]
        y_train.at[len(y_train) + 1] = synth_train_y
        
        train_normal = scaler.fit(X_train)
        X_train_normal = pd.DataFrame(train_normal.transform(X_train), columns = X_train.columns)
        X_test_normal = pd.DataFrame(train_normal.transform(X_test), columns = X_test.columns)
        
        svc.fit(X_train_normal, y_train)
        
        y_pred = svc.predict(X_test_normal)
        print(y_pred)
        y_test_list.append(y_test[0])
        
        y_pred_list.append(y_pred)
        
    #clean test list of series objects in order to calculate f1 metric properly
    for i in range(0, len(y_test_list)):
        if (i % 2 is not 0):
            filtered_test_list.append(y_test_list[i])
    
    print("YPRED: ")
    print(y_pred_list)
    print("YTEST: ")
    print(filtered_test_list)
    new_f1 = metrics.f1_score(filtered_test_list, y_pred_list)
    
    #find highest f1 score to compare new f1 score to
    for f1 in gan_acc_list:
        if f1 > threshold_f1:
            threshold_f1 = f1
        
    #print("THRESHOLD: " + str(threshold_f1))
    #print("NEW_f1:: " + str(new_f1))
    
    #if new f1 is higher than current max f1, append synthetic point to X set
    if new_f1 > threshold_f1:
        f1_change = new_f1 - threshold_f1
        #print("Synthetic sample added to container after f1 increase")
        synth_x_container = synth_x_container.append(synth_frame_x.loc[row])
        add_y = pd.Series(data = synth_train_y, index = None)
        synth_y_container = synth_y_container.append(add_y)
        synth_dict_key = str(row)
        synth_dict[synth_dict_key] = f1_change
    
    
    gan_acc_list.append(new_f1)
        
    #clear metric lists
    y_test_list = []
    filtered_test_list = []
    y_pred_list = []

#fix list:
    #synthetic data is not being appended to training dataframe correctly **FIXED**
    #clean up variable names for containers **IN PROGRESS**
    #use only cingulate region data

    
#version notes
#removed feature voting 
#removed unnecessary "rfe_features" argument from feature voting helper function
#fixed f1 score metric generation by adjusting y_test_list
#need to adjust the rest of the models to fix this function call
#removed recursive feature elimination
#removed optuna from training loop and replaced it with normal model generation, fitting and prediction