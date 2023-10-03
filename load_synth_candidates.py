import pandas as pd
import pickle 

synth_cand_x = pd.read_pickle("svm_cand_x.pkl")  
synth_cand_y = pd.read_pickle("svm_cand_y.pkl")

print(synth_cand_y)

