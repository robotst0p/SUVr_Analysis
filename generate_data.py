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

#establish initial frames for control and subject generation
synth_frame_x_1999 = pd.DataFrame(data = synthetic_suvr_1999[0], columns = aud_normal_x.columns)
synth_frame_x_29999 = pd.DataFrame(data = synthetic_suvr_29999[0], columns = aud_normal_x.columns)

synth_control_x_1999 = pd.DataFrame(data = synthetic_control_1999[0], columns = aud_normal_x.columns)
synth_control_x_29999 = pd.DataFrame(data = synthetic_control_29999[0], columns = aud_normal_x.columns)

def normalize_frame(frame):
    scaler2 = StandardScaler()

    synth_model = scaler2.fit(frame)
    synth_frame_normal = pd.DataFrame(synth_model.transform(frame), columns = frame.columns)

    return synth_frame_normal

#generate 100 sets of samples and controls, put them into one dataframe then average them out for each brain region
#normalize them at the end 
def add_samples(old_frame, type, iteration, new_frame = pd.DataFrame()):

    if iteration == 1999:
        generator = load_model('C:/Users/meyer/Desktop/SUVr_Analysis/weights/wgan_CingulateSUVR_1999.h5')
    else:
        generator = load_model('C:/Users/meyer/Desktop/SUVr_Analysis/weights/wgan_CingulateSUVR_29999.h5')

    #generate control data
    if type == 0:
        synth = gan.test_generator(generator, type, 16)
    else:
        synth = gan.test_generator(generator, type, 11)


    #place generated data in a dataframe
    new_frame = pd.DataFrame(data = synth[0], columns = old_frame.columns)
    #concatenate new generated data to old frame passed into function
    frame_list = [old_frame, new_frame]
    final_frame = pd.concat(frame_list)

    return final_frame

for i in range(0, 100):
    synth_frame_x_1999 = add_samples(synth_frame_x_1999, 1, 1999)
    synth_frame_x_29999 = add_samples(synth_frame_x_29999, 1, 29999)
    
    synth_control_x_1999 = add_samples(synth_control_x_1999, 0, 1999)
    synth_control_x_29999 = add_samples(synth_control_x_29999, 0, 29999)

# synth_frame_x_1999 = normalize_frame(synth_frame_x_1999)
# synth_frame_x_29999 = normalize_frame(synth_frame_x_29999)

# synth_control_x_1999 = normalize_frame(synth_control_x_1999)
# synth_control_x_29999 = normalize_frame(synth_control_x_29999)

print(synth_frame_x_1999)

#average out new frames 
    
#df.loc[:, 'weight'].mean()
average_synth_1999 = pd.DataFrame(columns = synth_frame_x_1999.columns)
average_synth_29999 = pd.DataFrame(columns = synth_frame_x_1999.columns)
average_control_1999 = pd.DataFrame(columns = synth_frame_x_1999.columns)
average_control_29999 = pd.DataFrame(columns = synth_frame_x_1999.columns)

for column in synth_frame_x_1999.columns:
    print(synth_frame_x_1999.loc[:, column].mean())
    average_synth_1999.loc[0, column] = synth_frame_x_1999.loc[:, column].mean()
    average_synth_29999.loc[0, column] = synth_frame_x_29999.loc[:, column].mean()
    average_control_1999.loc[0, column] = synth_control_x_1999.loc[:, column].mean()
    average_control_29999.loc[0, column] = synth_control_x_29999.loc[:, column].mean()

# average_synth_1999 = normalize_frame(average_synth_1999)
# average_synth_29999 = normalize_frame(average_synth_29999)

# average_control_1999 = normalize_frame(average_control_1999)
# average_control_29999 = normalize_frame(average_control_29999)

print(average_synth_29999)
#normalize data before pickling it
# synth_frame_x_1999 = pd.DataFrame(data = synthetic_suvr_1999[0], columns = aud_normal_x.columns)
# synth_control_x_1999 = pd.DataFrame(data = synthetic_control_1999[0], columns = aud_normal_x.columns)
# synth_frame_x_29999 = pd.DataFrame(data = synthetic_suvr_29999[0], columns = aud_normal_x.columns)
# synth_control_x_29999 = pd.DataFrame(data = synthetic_control_29999[0], columns = aud_normal_x.columns)

# #z-score normalization
# scaler2 = StandardScaler()

# #raw_data normalization of feature vector

# synth_suvr_model_1999 = scaler2.fit(synth_frame_x_1999)
# synth_suvr_model_29999 = scaler2.fit(synth_frame_x_29999)

# synth_control_model_1999 = scaler2.fit(synth_control_x_1999)
# synth_control_model_29999 = scaler2.fit(synth_control_x_29999)

# synth_suvr_normal_1999 = pd.DataFrame(synth_suvr_model_1999.transform(synth_frame_x_1999), columns = aud_normal_x.columns)
# synth_suvr_normal_29999 = pd.DataFrame(synth_suvr_model_29999.transform(synth_frame_x_29999), columns = aud_normal_x.columns)

# synth_control_model_1999 = scaler2.fit(synth_control_x_1999)
# synth_control_model_29999 = scaler2.fit(synth_control_x_29999)

# synth_control_normal_1999 = pd.DataFrame(synth_control_model_1999.transform(synth_control_x_1999), columns = aud_normal_x.columns)
# synth_control_normal_29999 = pd.DataFrame(synth_control_model_29999.transform(synth_control_x_29999), columns = aud_normal_x.columns)

#divergence_frame.to_pickle("./divergence_frame.pkl")
#save generated and normalized data frames for later analysis
#synth_suvr_normal_1999.to_pickle("./gen_suvr_1999")
#synth_suvr_normal_29999.to_pickle("./gen_suvr_29999")
#synth_control_normal_1999.to_pickle("./gen_control_1999")
#synth_control_normal_29999.to_pickle("./gen_control_29999")

#generate 100 sets of samples and controls, put them into one dataframe then average them out for each brain region
#normalize them at the end 