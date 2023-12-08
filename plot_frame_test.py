import pandas as pd
import pickle 
from tensorflow.keras.models import load_model
from lib import gan_aud as gan
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle
from scipy.special import rel_entr
from scipy.stats import entropy
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#load dataframe of kl divergence values 
divergence_frame = pd.read_pickle("./divergence_frame.pkl")  

row_names = {0:'ctx_lh_caudalanteriorcingulate',
             1:'ctx_lh_isthmuscingulate',
             2:'ctx_lh_posteriorcingulate',
             3:'ctx_lh_rostralanteriorcingulate',
             4:'ctx_rh_caudalanteriorcingulate',
             5:'ctx_rh_isthmuscingulate',
             6:'ctx_rh_posteriorcingulate',
             7:'ctx_rh_rostralanteriorcingulate'}

region_list = ['ctx_lh_caudalanteriorcingulate','ctx_lh_isthmuscingulate','ctx_lh_posteriorcingulate','ctx_lh_rostralanteriorcingulate','ctx_rh_caudalanteriorcingulate','ctx_rh_isthmuscingulate','ctx_rh_posteriorcingulate','ctx_rh_rostralanteriorcingulate']

divergence_frame = divergence_frame.rename(index = row_names)

print(divergence_frame)

#get first row of divergence frame
caudal_anteriorlh = divergence_frame.iloc[0,:].tolist()

plot_frame = pd.DataFrame(columns = ['KL Type','Region', 'Divergence'])

l = 0
k = 7
for column in divergence_frame.columns:
    for i in range (l, k + 1):
        plot_frame.loc[i,'KL Type'] = column

    l += 8
    k += 8
 
for i in range(0, len(plot_frame.index)):
    if i < 8:
        plot_frame.loc[i, 'Region'] = region_list[i]
    else:
        plot_frame.loc[i, 'Region'] = region_list[i % 8]

print(plot_frame)

