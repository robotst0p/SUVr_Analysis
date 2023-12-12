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

#original data no synthetic samples 
aud_normal_x = pd.read_pickle("./aud_frame_normal.pkl")  
control_normal_x = pd.read_pickle("./control_frame_normal.pkl")

synth_suvr_x_1999 = pd.read_pickle("./average_synth_1999.pkl")
synth_suvr_x_29999 = pd.read_pickle("./average_synth_29999.pkl")

synth_control_x_1999 = pd.read_pickle("./average_control_1999.pkl")
synth_control_x_29999 = pd.read_pickle("./average_control_29999.pkl")

pq_list = []
qp_list = []

#synthetic aud: synth_X_normal
#synthetic control: synth_control_normal
#original aud: aud_normal_x
#original control: control_normal_x

def kl_divergence(p, q):
        kl_pq = entropy(p, q)
        kl_qp = entropy(q, p)

        return kl_pq

def get_curve_data(axis, p_q):
        x, p = axis.get_lines()[p_q[0]].get_data()
        x, q = axis.get_lines()[p_q[1]].get_data()

        kl_pq = kl_divergence(p,q)

        return(kl_pq)

plt.clf()

axis_list = []

#lets start the analysis at 1999 iterations
f, ([ax1, ax2, ax3, ax4], [ax5, ax6, ax7, ax8]) = plt.subplots(nrows = 2, ncols = 4)
plt.title("HDAC/SUVR regional (Cingulate) value distribution (density plot)")

axis_list.append(ax1)
axis_list.append(ax2)
axis_list.append(ax3)
axis_list.append(ax4)
axis_list.append(ax5)
axis_list.append(ax6)
axis_list.append(ax7)
axis_list.append(ax8)

for i in range(0, len(axis_list)):
    sb.kdeplot(data = synth_control_x_29999.iloc[:,i].tolist(), ax = axis_list[i], label = 'synthetic CONTROL', color = 'cyan')

for i in range(0, len(axis_list)):
    sb.kdeplot(data = synth_suvr_x_29999.iloc[:,i].tolist(), ax = axis_list[i], label = 'synthetic AUD', color = 'orange')

for i in range(0, len(axis_list)):
    sb.kdeplot(data = aud_normal_x.iloc[:,i].tolist(), ax = axis_list[i], label = 'aud_original', color = 'red')

for i in range(0, len(axis_list)):
    sb.kdeplot(data = control_normal_x.iloc[:,i].tolist(), ax = axis_list[i], label = 'control_original', color = 'blue')

for axis in range(0, len(axis_list)):
        axis_list[axis].set(xlabel = aud_normal_x.columns[axis])

plt.draw()
plt.legend()
plt.show()

divergence_frame = pd.DataFrame(columns = ['synthaud_originalaud', 'synthcontrol_originalcontrol', 'synthaud_originalcontrol','synthcontrol_originalaud'])
#                                         [(1,2), (2,1), (0,3), (3,0), (1,3), (3,1), (1,0), (0,1), (0,2), (2,0)]
#                                         [(1,2),(0,3),(1,3),(0,2)]
#synth control: curve 0
#synthetic aud: curve 1
#aud original: curve 2
#control original: curve 3

curvelist = [(1,2), (0,3), (1,3), (0,2)]

for k in range(0, len(axis_list)):
    for i in range(0, len(divergence_frame.columns)):
        divergence_frame.loc[k, divergence_frame.columns[i]] = get_curve_data(axis_list[k], curvelist[i])


print(divergence_frame)
plt.clf()

divergence_frame.to_pickle("./divergence_frame_29999.pkl")

divergence_frame.to_csv("./divergence_frame_29999.csv")



