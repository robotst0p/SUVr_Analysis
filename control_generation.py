import pandas as pd
import pickle 
from tensorflow.keras.models import load_model
from lib import gan_aud as gan
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle
from scipy.special import rel_entr
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt

generator = load_model('C:/Users/meyer/Desktop/SUVr_Analysis/weights/wgan_CingulateSUVR_29999.h5')
synthetic_control = gan.test_generator(generator, 0)


scaler2 = StandardScaler()
raw_dataframe = pd.read_excel('AUD_SUVR_wb_cingulate.xlsx', index_col = 0)

#map subject labels to numerical values 
raw_dataframe.loc[raw_dataframe["CLASS"] == "AUD", "CLASS"] = 1
raw_dataframe.loc[raw_dataframe["CLASS"] == "CONTROL", "CLASS"] = 0

processed_data = raw_dataframe

control_frame = processed_data.loc[(processed_data['CLASS']) == 0]
aud_frame = processed_data.loc[(processed_data['CLASS']) == 1]

control_frame_x = control_frame.drop(['CLASS'], axis = 1)
aud_frame_x = aud_frame.drop(['CLASS'], axis = 1)

X_df = processed_data.drop(['CLASS'], axis = 1)

synth_frame_x = pd.DataFrame(data = synthetic_control[0], columns = X_df.columns)
synth_frame_y = pd.Series(synthetic_control[1])
#raw_data normalization of feature vector

X_model2 = scaler2.fit(synth_frame_x)
synth_X_normal = pd.DataFrame(X_model2.transform(synth_frame_x), columns = X_df.columns)
synth_X_normal.to_pickle("./synth_control_29999.pkl")