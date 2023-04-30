import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sn
import random
import time

 

#normalization
from sklearn.preprocessing import StandardScaler

#cpu training optimization for svm
from sklearnex import patch_sklearn
patch_sklearn()

#model/training importing 
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import RFE

 

#import LeaveOneOut for cross validation 
from sklearn.model_selection import LeaveOneOut

 

#metrics importing (accuracy, precision, sensitivity, recall)
from sklearn import metrics

#parameter tuning 
import optuna


#tensorflow for cgan model loading
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model

#from keras.utils.np_utils import to_categorical
from tensorflow.keras.utils import to_categorical

import tensorflow as tf
from keras.callbacks import TensorBoard

import numpy as np
import argparse
import pandas as pd
import datetime

from lib import gan_aud as gan

#load trained cgan
generator = load_model('/Users/tyler/Desktop/SUVr_Analysis/SUVr_Analysis/weights/wgan_CingulateSUVR_10499.h5')
synthetic_suvr = gan.test_generator(generator)

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
scaler1 = StandardScaler()

 

#raw_data normalization of feature vector
X_model = scaler1.fit(X)
X_normal = pd.DataFrame(X_model.transform(X), columns = X_df.columns)


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

 

svc = DecisionTreeClassifier()

 

cv = LeaveOneOut()

 


search_spaces =  { "criterion": optuna.distributions.CategoricalDistribution(choices = ('gini','entropy')), 
                  "splitter": optuna.distributions.CategoricalDistribution(choices = ('best','random')),
                  "max_depth": optuna.distributions.IntDistribution(1, 100),
                  "min_samples_split": optuna.distributions.IntDistribution(2,10)
                }

 

optuna_search = OptunaSearchCV(
    estimator=svc,
    param_distributions=search_spaces,
    n_trials=100,
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

#cgan connector notes:
    #use optuna before loop to fit best parameters for svm --> keep the same
    #use while loop with subject number threshold conditional on outside
    #take all preproccessing for synthetic data sets and place inside the loop
    #need to generate synth data, put into dataframe, normalize, then loop throw rows 
    
synth_counter = 0

while synth_counter <= 27:
    synthetic_suvr = gan.test_generator(generator)
    
    synth_frame_x = pd.DataFrame(data = synthetic_suvr[0], columns = X_df.columns)
    synth_frame_y = pd.Series(synthetic_suvr[1])
    
    #z-score normalization
    scaler2 = StandardScaler()

    #raw_data normalization of feature vector
    X_model2 = scaler2.fit(synth_frame_x)
    synth_X_normal = pd.DataFrame(X_model2.transform(synth_frame_x), columns = X_df.columns)
    
    for row in list(synth_X_normal.index.values):
        y_pred_list=[]
        svc = DecisionTreeClassifier(criterion = 'entropy', splitter = 'random', max_depth = 12, min_samples_split = 6)    
    
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
            #y_test_list.append(y_test[0])
            
            y_pred_list.append(y_pred[0])
            
        
        y_pred_final = pd.Series(y_pred_list)
        
        # print("YPRED: ")
        # print(y_pred_list)
        del svc
        
        #score = metrics.f1_score(y, y_pred_final)
        score = metrics.accuracy_score(y, y_pred_final)
    
     
        
        #find highest f1 score to compare new f1 score to
        if score > current_highest_score:
            current_highest_score = score
            #X_train_intermediate
            succesful_cand_X = succesful_cand_X.append(synth_cand_x)
            succesful_cand_Y = succesful_cand_Y.append(y_train_intermediate[-1:])
            
            print("ACCURACY INCREASED, SYNTHETIC CANDIDATE ADDED")
            print("NEW ACCURACY: " + str(score))
            print(synth_cand_x)
            time.sleep(1)
            
            synth_counter += 1

