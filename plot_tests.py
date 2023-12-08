from lib import gan_aud as gan
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.preprocessing import StandardScaler
from matplotlib.lines import Line2D
import math
from decimal import Decimal
from scipy.special import rel_entr
import numpy as np


aud_normal_x = pd.read_pickle("./aud_frame_normal.pkl")  
control_normal_x = pd.read_pickle("./control_frame_normal.pkl")
svm_x = pd.read_pickle("./svm_cand_x.pkl")

plt.clf()
f, ([ax1, ax2, ax3, ax4], [ax5, ax6, ax7, ax8]) = plt.subplots(nrows = 2, ncols = 4)

p = svm_x.iloc[:,0].tolist()
q = aud_normal_x.iloc[:,0].tolist()
print(p)
print(q)

p = [0, 6, 0, .4]
q = [.1818, .4545, .0909, .2726]

print(p)
print(q)

kl_pq = sum(rel_entr(q, p))
print(kl_pq)

#plt.title("HDAC/SUVR regional (Cingulate) value distribution (density plot)")

#sb.kdeplot(data = svm_x.iloc[:,0], ax = ax1, label = 'synthetic AUD', color = 'orange')
#sb.kdeplot(data = svm_x.iloc[:,1], ax = ax2, label = 'synthetic AUD', color = 'orange')
#sb.kdeplot(data = svm_x.iloc[:,2], ax = ax3, label = 'synthetic AUD', color = 'orange')
#sb.kdeplot(data = svm_x.iloc[:,3], ax = ax4, label = 'synthetic AUD', color = 'orange')
#sb.kdeplot(data =svm_x.iloc[:,4], ax = ax5, label = 'synthetic AUD', color = 'orange')
#sb.kdeplot(data =svm_x.iloc[:,5], ax = ax6, label = 'synthetic AUD', color = 'orange')
#sb.kdeplot(data =svm_x.iloc[:,6], ax = ax7, label = 'synthetic AUD', color = 'orange')
#sb.kdeplot(data =svm_x.iloc[:,7], ax = ax8, label = 'synthetic AUD', color = 'orange')


#sb.kdeplot(data = aud_normal_x.iloc[:, 0], ax = ax1, label = 'aud_original', color = 'red')
#sb.kdeplot(data = aud_normal_x.iloc[:, 1], ax = ax2, label = 'aud_original', color = 'red')
#sb.kdeplot(data = aud_normal_x.iloc[:, 2], ax = ax3, label = 'aud_original', color = 'red')
#sb.kdeplot(data = aud_normal_x.iloc[:, 3], ax = ax4, label = 'aud_original', color = 'red')
#sb.kdeplot(data = aud_normal_x.iloc[:, 4], ax = ax5, label = 'aud_original', color = 'red')
#sb.kdeplot(data = aud_normal_x.iloc[:, 5], ax = ax6, label = 'aud_original', color = 'red')
#sb.kdeplot(data = aud_normal_x.iloc[:, 6], ax = ax7, label = 'aud_original', color = 'red')
#sb.kdeplot(data = aud_normal_x.iloc[:, 7], ax = ax8, label = 'aud_original', color = 'red')

#sb.kdeplot(data = control_normal_x.iloc[:, 0], ax = ax1, label = 'control_original', color = 'blue')
#sb.kdeplot(data = control_normal_x.iloc[:, 1], ax = ax2, label = 'control_original', color = 'blue')
#sb.kdeplot(data = control_normal_x.iloc[:, 2], ax = ax3, label = 'control_original', color = 'blue')
#sb.kdeplot(data = control_normal_x.iloc[:, 3], ax = ax4, label = 'control_original', color = 'blue')
#sb.kdeplot(data = control_normal_x.iloc[:, 4], ax = ax5, label = 'control_original', color = 'blue')
#sb.kdeplot(data = control_normal_x.iloc[:, 5], ax = ax6, label = 'control_original', color = 'blue')
#sb.kdeplot(data = control_normal_x.iloc[:, 6], ax = ax7, label = 'control_original', color = 'blue')
#sb.kdeplot(data = control_normal_x.iloc[:, 7], ax = ax8, label = 'control_original', color = 'blue')

#plt.draw()
#plt.legend()
#plt.show()
#plt.clf()

#needs to be done:
#change plot color scheme
#create part 2 classification plots with three curves
#rerun metrics and recreate accuracy, sensitivity and specificity graphs
#reread gan paper and research unet architecture
#implement unet architecture for 3d image brain scan network
#research and implement genetic algorithm for feature selection in dlb data
