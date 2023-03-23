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

#helping functions 
from helper_functions import retrieve_feature_names, feature_vote

#parameter tuning 
import optuna

#load in suvr data as pandas dataframe
raw_dataframe = pd.read_excel('AUD_SUVr_wb_cingulate.xlsx', index_col = 0)

#map subject labels to numerical values 
raw_dataframe.loc[raw_dataframe["CLASS"] == "AUD", "CLASS"] = 1
raw_dataframe.loc[raw_dataframe["CLASS"] == "CONTROL", "CLASS"] = 0

processed_data = raw_dataframe

X_df = processed_data.drop(['CLASS'], axis = 1)

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

#fake matrix to test GAN loop structure
synth_frame_x = X_df.copy()
synth_frame_y = raw_dataframe['CLASS']

column_names = X_df.columns

synth_class_list = []
    
#fill fake matrix with random values
for row in list(synth_frame_x.index.values):
    synth_point_list = [random.random()]*100
    for column in column_names:
        index = random.randint(0, len(synth_point_list) - 1)
        synth_frame_x.loc[row, str(column)] = synth_point_list[index]
        
for i in range(0, len(synth_frame_y)):
    synth_class = random.choice([0,1])
    synth_frame_y[i] = synth_class
    
#rename row labels for synth dataset
synth_row_labels = {'AUD_A_006':'synth_0',
                    'AUD_A_009':'synth_1',
                    'AUD_A_016':'synth_2',
                    'AUD_A_018':'synth_3',
                    'aud_a_019':'synth_4',
                    'AUD_A_021':'synth_5',
                    'AUD_A_022':'synth_6',
                    'AUD_A_026':'synth_7',
                    'AUD_A_027':'synth_8',
                    'AUD_P_002':'synth_9',
                    'AUD_P_003':'synth_10',
                    'MSTAT_006_01':'synth_11',
                    'AUD_C_001':'synth_12',
                    'MSTAT_002_01':'synth_13',
                    'MSTAT_001_01':'synth_14',
                    'AUD_C_020':'synth_15',
                    'ADCON_012':'synth_16',
                    'MCON_002':'synth_17',
                    'AUD_C_005':'synth_18',
                    'aud_c_013':'synth_19',
                    'MCON_014':'synth_20',
                    'AUD_C_009':'synth_21',
                    'MSTAT_004_01':'synth_22',
                    'AUD_C_003':'synth_23',
                    'AUD_C_017':'synth_24',
                    'AUD_C_024':'synth_25',
                    'ADCON_002':'synth_26'}

synth_frame_x.rename(index = synth_row_labels, inplace = True)
    
synth_frame_y = synth_frame_y.astype(int)

synth_frame_y.rename(index = synth_row_labels, inplace = True)

feature_voting_list = []
final_feature_list = []

threshold_f1 = 0

y_test_list = []
filtered_test_list = []
y_pred_list = []

for row in list(synth_frame_x.index.values):
    for train_index, test_index in loo.split(X):
        svc = svm.SVC(kernel = "rbf", random_state = 42)    
        
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        y_test_list.append(y_test)
        
        #append synthetic point to X_training set
        testing_x = synth_frame_x.loc[row]
        X_train = X_train.append(testing_x, ignore_index = True)
        
        #append synthetic point to y_training set
        testing_y = synth_frame_y.loc[row]
        y_train.at[len(y_train) + 1] = testing_y
        
        train_normal = scaler.fit(X_train)
        X_train_normal = pd.DataFrame(train_normal.transform(X_train), columns = X_train.columns)
        X_test_normal = pd.DataFrame(train_normal.transform(X_test), columns = X_test.columns)
        
        svc.fit(X_train_normal, y_train)
        
        y_pred = svc.predict(X_test_normal)
        y_test_list.append(y_test[0])
        
        y_pred_list.append(y_pred)
        
    #clean test list of series objects
    for i in range(0, len(y_test_list)):
        if (i % 2 is not 0):
            filtered_test_list.append(y_test_list[i])
    
    
    new_f1 = metrics.f1_score(y_pred_list, filtered_test_list)
    print(new_f1)
    
    #find highest f1 score to compare new f1 score to
    for f1 in gan_acc_list:
        if f1 > threshold_f1:
            threshold_f1 = f1
        
    #if new f1 is higher than current max f1, append synthetic point to X set
    if new_f1 > threshold_f1:
        print("Synthetic sample added to training set")
        synth_x_container = synth_x_container.append(synth_frame_x.loc[row])
        print("Synthetic x added to training set")
        add_y = pd.Series(data = testing_y, index = None)
        synth_y_container = synth_y_container.append(add_y)
        print("Synthetic y added to training set")
    
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