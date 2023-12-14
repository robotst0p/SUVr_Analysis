import pandas as pd
import pickle 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from matplotlib import markers

divergence_frame = pd.read_pickle("./whole_divergence_frame.pkl")

print(divergence_frame)