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
scaler1 = StandardScaler()

 

#raw_data normalization of feature vector
X_model = scaler1.fit(X)
X_normal = pd.DataFrame(X_model.transform(X), columns = X_df.columns)

 


#z-score normalization
scaler2 = StandardScaler()

 

#raw_data normalization of feature vector
X_model2 = scaler2.fit(synth_frame_x)
synth_X_normal = pd.DataFrame(X_model2.transform(synth_frame_x), columns = X_df.columns)

 

 

#leaveoneout cross validation and decisiontree model creation 
loo = LeaveOneOut()
loo.get_n_splits(X)

 

 

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

 


from optuna.integration import OptunaSearchCV
from optuna.distributions import CategoricalDistribution, LogUniformDistribution

 

svc = svm.SVC(kernel='rbf', random_state=42)

 

cv = LeaveOneOut()

 


search_spaces =  { "C": optuna.distributions.FloatDistribution(0.1, 10, step=0.1)
                }

 

optuna_search = OptunaSearchCV(
    estimator=svc,
    param_distributions=search_spaces,
    n_trials=20,
    cv=cv,
    error_score=0.0,
    refit=True,
)

 


optuna_search.fit(X_normal, y.astype(int))

 

optuna_search.best_score_
optuna_search.best_params_

 

current_highest_score= optuna_search.best_score_

 


succesful_cand_X=pd.DataFrame()#pd.dataframe
succesful_cand_Y=pd.Series()

 

for row in list(synth_X_normal.index.values):
    y_pred_list=[]
    svc = svm.SVC(kernel = "rbf", C=1.7, random_state = 42)    

 

    for train_index, test_index in loo.split(X_normal):
        
        X_train, X_test = X_normal.iloc[train_index], X_normal.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        #y_test_list.append(y_test)
        
        #append synthetic point to X_training set
        synth_cand_x = synth_X_normal.loc[row]
        
        X_train_intermediate = X_train.append(synth_cand_x, ignore_index = True) # xtrain intermediate is a orig+synthetic data
        
        #append synthetic point to y_training set
        synth_train_y = synth_frame_y.loc[row]
        
        y_train_intermediate=y_train.copy()
        
        y_train_intermediate.at[len(y_train) + 1] = synth_train_y
        
        #y_train=y_train.rename({27: row})
        
        y_train_intermediate = y_train_intermediate.rename({len(y_train_intermediate):row})
        #####################################################
        ############### change after decision 
        X_train_intermediate = X_train_intermediate.append(succesful_cand_X)
        y_train_intermediate = y_train_intermediate.append(succesful_cand_Y)
        
        svc.fit(X_train_intermediate, y_train_intermediate)
        
        y_pred = svc.predict(X_test)
        print(y_pred)
        #y_test_list.append(y_test[0])
        
        y_pred_list.append(y_pred[0])
        
    
    y_pred_final = pd.Series(y_pred_list)
    
    # print("YPRED: ")
    # print(y_pred_list)
    del svc
    print("ROW evaluation: ")
    print(row)
    
    #score = metrics.f1_score(y, y_pred_final)
    score = metrics.accuracy_score(y, y_pred_final)

 

    
    #find highest f1 score to compare new f1 score to
    if score > current_highest_score:
        current_highest_score = score
        #X_train_intermediate
        succesful_cand_X = succesful_cand_X.append(synth_cand_x)
        succesful_cand_Y = succesful_cand_Y.append(y_train_intermediate[-1:])

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