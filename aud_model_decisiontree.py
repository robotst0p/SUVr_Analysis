import pandas as pd
import numpy 

#model/training imports
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.decomposition import PCA

#metrics importing (precision, accuracy, recall, sensitivity, confusion matrix)
from sklearn import metrics 

aud_data = pd.read_excel('AUD_SUVr_WB.xlsx', index_col = 0)

train, test = train_test_split(aud_data, test_size = 0.4, stratify = aud_data['CLASS'], random_state = 42)

X_train = train.drop(['CLASS'], axis = 1)
y_train = train.CLASS
X_test = test.drop(['CLASS'], axis = 1)
y_test = test.CLASS

#apply pca for feature selection
pca = PCA(n_components = 5)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
explained_variance = pca.explained_variance_ratio_
print(explained_variance)

#build classification model
mod_dt = DecisionTreeClassifier(max_depth = 3, random_state = 1)
mod_dt.fit(X_train, y_train)
prediction = mod_dt.predict(X_test)
print('The accuracy of the Decision Tree is', "{:.3f}".format(metrics.accuracy_score(prediction, y_test)))