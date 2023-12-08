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
import seaborn as sb
import matplotlib.pyplot as plt

#load original dataframe as reference
aud_normal_x = pd.read_pickle("./aud_frame_normal.pkl")  

#generator trained for 1999 iterations
generator_1999 = load_model('C:/Users/meyer/Desktop/SUVr_Analysis/weights/wgan_CingulateSUVR_1999.h5')

#generator trained for 29999 iterations
generator_29999 = load_model('C:/Users/meyer/Desktop/SUVr_Analysis/weights/wgan_CingulateSUVR_29999.h5')

#generate subject data for 1999 iterations
synthetic_suvr_1999 = gan.test_generator(generator_1999, 1, 11)

#generate control data for 1999 iterations
synthetic_control_1999 = gan.test_generator(generator_1999, 0, 16)

#generate synthetic subject data for 29999 iterations
synthetic_suvr_29999 = gan.test_generator(generator_29999, 1, 11)

#generate synthetic subject data for 29999 iterations
synthetic_control_29999 = gan.test_generator(generator_29999, 0, 16)

#normalize data before pickling it
synth_frame_x_1999 = pd.DataFrame(data = synthetic_suvr_1999[0], columns = aud_normal_x.columns)
synth_control_x_1999 = pd.DataFrame(data = synthetic_control_1999[0], columns = aud_normal_x.columns)
synth_frame_x_29999 = pd.DataFrame(data = synthetic_suvr_29999[0], columns = aud_normal_x.columns)
synth_control_x_29999 = pd.DataFrame(data = synthetic_control_29999[0], columns = aud_normal_x.columns)

#z-score normalization
scaler2 = StandardScaler()

#raw_data normalization of feature vector

synth_suvr_model_1999 = scaler2.fit(synth_frame_x_1999)
synth_suvr_model_29999 = scaler2.fit(synth_frame_x_29999)

synth_control_model_1999 = scaler2.fit(synth_control_x_1999)
synth_control_model_29999 = scaler2.fit(synth_control_x_29999)

synth_suvr_normal_1999 = pd.DataFrame(synth_suvr_model_1999.transform(synth_frame_x_1999), columns = aud_normal_x.columns)
synth_suvr_normal_29999 = pd.DataFrame(synth_suvr_model_29999.transform(synth_frame_x_29999), columns = aud_normal_x.columns)

synth_control_model_1999 = scaler2.fit(synth_control_x_1999)
synth_control_model_29999 = scaler2.fit(synth_control_x_29999)

synth_control_normal_1999 = pd.DataFrame(synth_control_model_1999.transform(synth_control_x_1999), columns = aud_normal_x.columns)
synth_control_normal_29999 = pd.DataFrame(synth_control_model_29999.transform(synth_control_x_29999), columns = aud_normal_x.columns)

#divergence_frame.to_pickle("./divergence_frame.pkl")
#save generated and normalized data frames for later analysis
synth_suvr_normal_1999.to_pickle("./gen_suvr_1999")
synth_suvr_normal_29999.to_pickle("./gen_suvr_29999")
synth_control_normal_1999.to_pickle("./gen_control_1999")
synth_control_normal_29999.to_pickle("./gen_control_29999")

