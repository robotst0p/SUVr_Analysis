import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
plt.matplotlib.use('TkAgg')
import seaborn as sn
import random
import time

#disable pandas warnings for deprecated methods 
import warnings
warnings.filterwarnings('ignore')

#density plotting for generated features
import seaborn as sb

#normalization
from sklearn.preprocessing import StandardScaler

#cpu training optimization for svm
from sklearnex import patch_sklearn
patch_sklearn()

#model/training importing 
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

#import LeaveOneOut for cross validation 
from sklearn.model_selection import LeaveOneOut

#metrics importing (accuracy, precision, sensitivity, recall)
from sklearn import metrics

#parameter tuning 
import optuna

#tensorflow for cgan model loading
from tensorflow.keras.models import load_model
import tensorflow as tf

import numpy as np
import argparse
import pandas as pd
import datetime

#save and load variables
import pickle

#import generator model
from lib import gan_aud as gan

#load trained cgan
generator = load_model('C:/Users/meyer/Desktop/SUVr_Analysis/weights/wgan_CingulateSUVR_29999.h5')
synthetic_suvr = gan.test_generator(generator)


#load in suvr data or only cingulate data as pandas dataframe
raw_dataframe = pd.read_excel('AUD_SUVR_wb_cingulate.xlsx', index_col = 0)

#map subject labels to numerical values 
raw_dataframe.loc[raw_dataframe["CLASS"] == "AUD", "CLASS"] = 1
raw_dataframe.loc[raw_dataframe["CLASS"] == "CONTROL", "CLASS"] = 0

#keep original raw data intact in case we want to use later
processed_data = raw_dataframe

control_frame = processed_data.loc[(processed_data['CLASS']) == 0]
aud_frame = processed_data.loc[(processed_data['CLASS']) == 1]

control_frame_x = control_frame.drop(['CLASS'], axis = 1)
aud_frame_x = aud_frame.drop(['CLASS'], axis = 1)

X_df = processed_data.drop(['CLASS'], axis = 1)

#convert to numpy array for training 
X = X_df.copy()
control_x = control_frame_x.copy()
aud_x = aud_frame_x.copy()

y = raw_dataframe['CLASS']
y = y.astype(int)

synth_y_container = pd.Series()
synth_x_container = pd.DataFrame(columns = X.columns)

#z-score normalization
scaler1 = StandardScaler()

#raw_data normalization of feature vector
X_model = scaler1.fit(X)
control_x_model = scaler1.fit(control_x)
aud_x_model = scaler1.fit(aud_x)

X_normal = pd.DataFrame(X_model.transform(X), columns = X_df.columns)
control_frame_normal = pd.DataFrame(control_x_model.transform(control_x), columns = X_df.columns)
aud_frame_normal = pd.DataFrame(aud_x_model.transform(aud_x), columns = X_df.columns)

#save the original dataframes for later comparison
aud_frame_normal.to_pickle("./aud_frame_normal.pkl")
control_frame_normal.to_pickle("./control_frame_normal.pkl")

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

#prompt user to enter desired model type
print("MODELS:")
print("svm" + "\n" + "xgboost" + "\n" + "decisiontree" + "\n" + "randomforest") 

model_select = input("ENTER DESIRED MODEL OF THOSE LISTED: ")

if (model_select == "svm"):
    reset_model = svm.SVC(kernel='rbf', random_state=42)
elif (model_select == "randomforest"):
    reset_model = RandomForestClassifier()
elif (model_select == "xgboost"):
    reset_model = xgb.XGBClassifier()
elif (model_select == "decisiontree"):
    reset_model = DecisionTreeClassifier()

cv = LeaveOneOut()

#optuna search parameters for each model type
search_spaces_svm =  { "C": optuna.distributions.FloatDistribution(0.1, 10, step=0.1)
                }

search_spaces_logreg =  { "C": optuna.distributions.FloatDistribution(0.1, 10, step=0.1),
                         "penalty": optuna.distributions.CategoricalDistribution(["l1","l2","elasticnet"]),
                         "dual": optuna.distributions.CategoricalDistribution([True,False]),
                         "tol": optuna.distributions.FloatDistribution(0, .1, step = .1),
                         "random_state": optuna.distributions.IntDistribution(1, 100, step = 1),
                         "solver": optuna.distributions.CategoricalDistribution(["lbfgs","liblinear","newton-cg","newton-cholesky","sag","saga"])
                }

search_spaces_decisiontree =  { "criterion": optuna.distributions.CategoricalDistribution(["gini","entropy","log_loss"]),
                               "splitter": optuna.distributions.CategoricalDistribution(["best","random"]),
                               "max_depth": optuna.distributions.IntDistribution(1, 100, step = 1),
                               "min_samples_split": optuna.distributions.IntDistribution(2, 10, step = 1),
                               "random_state": optuna.distributions.IntDistribution(0, 100, step = 5)

                }

search_spaces_randomforest =  { "criterion": optuna.distributions.CategoricalDistribution(["gini","entropy"]),
                                "n_estimators": optuna.distributions.IntDistribution(100, 1000, step = 100),
                                "max_depth": optuna.distributions.IntDistribution(1, 100, step = 1),
                                "min_samples_split": optuna.distributions.IntDistribution(2, 10, step = 1)
                }

search_spaces_xgboost = {"tree_method": optuna.distributions.CategoricalDistribution(["auto","gpu_hist"])}
 
if (model_select == "svm"):
    param_select = search_spaces_svm
elif (model_select == "randomforest"):
    param_select = search_spaces_randomforest
elif (model_select == "xgboost"):
    param_select = search_spaces_svm
elif (model_select == "decisiontree"):
    param_select = search_spaces_decisiontree

optuna_search = OptunaSearchCV(
    estimator=reset_model,
    param_distributions = param_select,
    n_trials=20,
    cv=cv,
    error_score=0.0,
    refit=True,
)

optuna_search.fit(X_normal, y.astype(int))

optuna_search.best_score_
best_params = optuna_search.best_params_

if (model_select == "svm"):
    reset_model = svm.SVC(**best_params)
elif (model_select == "randomforest"):
    reset_model = RandomForestClassifier(**best_params)
elif (model_select == "xgboost"):
    reset_model = xgb.XGBClassifier()
elif (model_select == "decisiontree"):
    reset_model = DecisionTreeClassifier(**best_params)

print(best_params) 

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
    #{'criterion': 'entropy', 'n_estimators': 500, 'max_depth': 24, 'min_samples_split': 4}.
    for row in list(synth_X_normal.index.values):
        y_pred_list=[]
        y_test_list=[] 
        svc = reset_model  
    
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
            y_test_list.append(y_test[0])
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
            succesful_cand_X.to_pickle("./randomforest_cand_x.pkl")
            succesful_cand_Y.to_pickle("./randomforest_cand_y.pkl")

            plt.clf()
            f, ([ax1, ax2, ax3, ax4], [ax5, ax6, ax7, ax8]) = plt.subplots(nrows = 2, ncols = 4)

            plt.title("HDAC/SUVR regional (Cingulate) value distribution (density plot)")

            sb.kdeplot(data = succesful_cand_X.iloc[:,0], ax = ax1, label = 'synthetic AUD', color = 'orange')
            sb.kdeplot(data = succesful_cand_X.iloc[:,1], ax = ax2, label = 'synthetic AUD', color = 'orange')
            sb.kdeplot(data = succesful_cand_X.iloc[:,2], ax = ax3, label = 'synthetic AUD', color = 'orange')
            sb.kdeplot(data = succesful_cand_X.iloc[:,3], ax = ax4, label = 'synthetic AUD', color = 'orange')
            sb.kdeplot(data = succesful_cand_X.iloc[:,4], ax = ax5, label = 'synthetic AUD', color = 'orange')
            sb.kdeplot(data = succesful_cand_X.iloc[:,5], ax = ax6, label = 'synthetic AUD', color = 'orange')
            sb.kdeplot(data = succesful_cand_X.iloc[:,6], ax = ax7, label = 'synthetic AUD', color = 'orange')
            sb.kdeplot(data = succesful_cand_X.iloc[:,7], ax = ax8, label = 'synthetic AUD', color = 'orange')

            sb.kdeplot(data = aud_frame_normal.iloc[:, 0], ax = ax1, label = 'aud_original', color = 'red')
            sb.kdeplot(data = aud_frame_normal.iloc[:, 1], ax = ax2, label = 'aud_original', color = 'red')
            sb.kdeplot(data = aud_frame_normal.iloc[:, 2], ax = ax3, label = 'aud_original', color = 'red')
            sb.kdeplot(data = aud_frame_normal.iloc[:, 3], ax = ax4, label = 'aud_original', color = 'red')
            sb.kdeplot(data = aud_frame_normal.iloc[:, 4], ax = ax5, label = 'aud_original', color = 'red')
            sb.kdeplot(data = aud_frame_normal.iloc[:, 5], ax = ax6, label = 'aud_original', color = 'red')
            sb.kdeplot(data = aud_frame_normal.iloc[:, 6], ax = ax7, label = 'aud_original', color = 'red')
            sb.kdeplot(data = aud_frame_normal.iloc[:, 7], ax = ax8, label = 'aud_original', color = 'red')

            sb.kdeplot(data = control_frame_normal.iloc[:, 0], ax = ax1, label = 'control_original', color = 'blue')
            sb.kdeplot(data = control_frame_normal.iloc[:, 1], ax = ax2, label = 'control_original', color = 'blue')
            sb.kdeplot(data = control_frame_normal.iloc[:, 2], ax = ax3, label = 'control_original', color = 'blue')
            sb.kdeplot(data = control_frame_normal.iloc[:, 3], ax = ax4, label = 'control_original', color = 'blue')
            sb.kdeplot(data = control_frame_normal.iloc[:, 4], ax = ax5, label = 'control_original', color = 'blue')
            sb.kdeplot(data = control_frame_normal.iloc[:, 5], ax = ax6, label = 'control_original', color = 'blue')
            sb.kdeplot(data = control_frame_normal.iloc[:, 6], ax = ax7, label = 'control_original', color = 'blue')
            sb.kdeplot(data = control_frame_normal.iloc[:, 7], ax = ax8, label = 'control_original', color = 'blue')

            plt.draw()
            plt.legend()
            plt.show()
            plt.clf()
            #plt.clf()
            #sb.kdeplot(data=X_normal.iloc[:, 1])
            #sb.kdeplot(data=succesful_cand_X.iloc[:,1])
            print("ACCURACY INCREASED, SYNTHETIC CANDIDATE ADDED")
            print("NEW ACCURACY: " + str(score))
            print(synth_cand_x)
            
            print("Accuracy:", metrics.accuracy_score(y_test_list, y_pred_list))
            print("Precision:", metrics.precision_score(y_test_list, y_pred_list))
            print("Recall:", metrics.recall_score(y_test_list, y_pred_list))
            
            print("METRICS REPORT:", metrics.classification_report(y_test_list, y_pred_list))
            
            time.sleep(1)
            
            synth_counter += 1

#version notes
#removed extra library imports
#removed leftover synthetic data preprocessing 