from lib import gan_aud as gan
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

aud_normal_x = pd.read_pickle("./aud_frame_normal.pkl")  
control_normal_x = pd.read_pickle("./control_frame_normal.pkl")

generator = load_model('C:/Users/meyer/Desktop/SUVr_Analysis/weights/wgan_CingulateSUVR_1999.h5')
synthetic_suvr_con = gan.test_generator(generator, 0)
synthetic_suvr_aud = gan.test_generator(generator, 1)

synth_con_frame = pd.DataFrame(data = synthetic_suvr_con[0], columns = aud_normal_x.columns)
synth_aud_frame = pd.DataFrame(data = synthetic_suvr_aud[0], columns = aud_normal_x.columns)

scaler1 = StandardScaler()
scaler2 = StandardScaler()

X_model1 = scaler1.fit(synth_aud_frame)
X_model2 = scaler2.fit(synth_con_frame)

synth_aud_normal = pd.DataFrame(X_model1.transform(synth_aud_frame), columns = aud_normal_x.columns)
synth_con_normal = pd.DataFrame(X_model2.transform(synth_con_frame), columns = aud_normal_x.columns)


plt.clf()
f, ([ax1, ax2, ax3, ax4], [ax5, ax6, ax7, ax8]) = plt.subplots(nrows = 2, ncols = 4)

plt.title("HDAC/SUVR regional (Cingulate) value distribution (density plot)")

sb.kdeplot(data = synth_aud_normal.iloc[:,0], ax = ax1, label = 'synthetic AUD', color = 'green')
sb.kdeplot(data = synth_aud_normal.iloc[:,1], ax = ax2, label = 'synthetic AUD', color = 'green')
sb.kdeplot(data = synth_aud_normal.iloc[:,2], ax = ax3, label = 'synthetic AUD', color = 'green')
sb.kdeplot(data = synth_aud_normal.iloc[:,3], ax = ax4, label = 'synthetic AUD', color = 'green')
sb.kdeplot(data =synth_aud_normal.iloc[:,4], ax = ax5, label = 'synthetic AUD', color = 'green')
sb.kdeplot(data =synth_aud_normal.iloc[:,5], ax = ax6, label = 'synthetic AUD', color = 'green')
sb.kdeplot(data =synth_aud_normal.iloc[:,6], ax = ax7, label = 'synthetic AUD', color = 'green')
sb.kdeplot(data =synth_aud_normal.iloc[:,7], ax = ax8, label = 'synthetic AUD', color = 'green')

sb.kdeplot(data = aud_normal_x.iloc[:, 0], ax = ax1, label = 'aud_original', color = 'red')
sb.kdeplot(data = aud_normal_x.iloc[:, 1], ax = ax2, label = 'aud_original', color = 'red')
sb.kdeplot(data = aud_normal_x.iloc[:, 2], ax = ax3, label = 'aud_original', color = 'red')
sb.kdeplot(data = aud_normal_x.iloc[:, 3], ax = ax4, label = 'aud_original', color = 'red')
sb.kdeplot(data = aud_normal_x.iloc[:, 4], ax = ax5, label = 'aud_original', color = 'red')
sb.kdeplot(data = aud_normal_x.iloc[:, 5], ax = ax6, label = 'aud_original', color = 'red')
sb.kdeplot(data = aud_normal_x.iloc[:, 6], ax = ax7, label = 'aud_original', color = 'red')
sb.kdeplot(data = aud_normal_x.iloc[:, 7], ax = ax8, label = 'aud_original', color = 'red')

sb.kdeplot(data = control_normal_x.iloc[:, 0], ax = ax1, label = 'control_original', color = 'yellow')
sb.kdeplot(data = control_normal_x.iloc[:, 1], ax = ax2, label = 'control_original', color = 'yellow')
sb.kdeplot(data = control_normal_x.iloc[:, 2], ax = ax3, label = 'control_original', color = 'yellow')
sb.kdeplot(data = control_normal_x.iloc[:, 3], ax = ax4, label = 'control_original', color = 'yellow')
sb.kdeplot(data = control_normal_x.iloc[:, 4], ax = ax5, label = 'control_original', color = 'yellow')
sb.kdeplot(data = control_normal_x.iloc[:, 5], ax = ax6, label = 'control_original', color = 'yellow')
sb.kdeplot(data = control_normal_x.iloc[:, 6], ax = ax7, label = 'control_original', color = 'yellow')
sb.kdeplot(data = control_normal_x.iloc[:, 7], ax = ax8, label = 'control_original', color = 'yellow')

sb.kdeplot(data = synth_con_normal.iloc[:, 0], ax = ax1, label = 'synthetic CONTROL', color = 'purple')
sb.kdeplot(data = synth_con_normal.iloc[:, 1], ax = ax2, label = 'synthetic CONTROL', color = 'purple')
sb.kdeplot(data = synth_con_normal.iloc[:, 2], ax = ax3, label = 'synthetic CONTROL', color = 'purple')
sb.kdeplot(data = synth_con_normal.iloc[:, 3], ax = ax4, label = 'synthetic CONTROL', color = 'purple')
sb.kdeplot(data = synth_con_normal.iloc[:, 4], ax = ax5, label = 'synthetic CONTROL', color = 'purple')
sb.kdeplot(data = synth_con_normal.iloc[:, 5], ax = ax6, label = 'synthetic CONTROL', color = 'purple')
sb.kdeplot(data = synth_con_normal.iloc[:, 6], ax = ax7, label = 'synthetic CONTROL', color = 'purple')
sb.kdeplot(data = synth_con_normal.iloc[:, 7], ax = ax8, label = 'synthetic CONTROL', color = 'purple')

plt.draw()
plt.legend()
plt.show()
plt.clf()

#needs to be done:
#change plot color scheme
#create part 2 classification plots with three curves
#rerun metrics and recreate accuracy, sensitivity and specificity graphs
#reread gan paper and research unet architecture
#implement unet architecture for 3d image brain scan network
#research and implement genetic algorithm for feature selection in dlb data
