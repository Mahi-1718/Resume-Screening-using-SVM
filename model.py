import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings('ignore')
from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score
from pandas.plotting import scatter_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm

data = pd.read_csv('newresume.csv')
requiredText = data['cleaned_resume'].values
requiredTarget = data['Category'].values



cv = CountVectorizer()
vv = cv.fit_transform(requiredText)



X_train,X_test,y_train,y_test = train_test_split(vv,requiredTarget,random_state=42, test_size=0.2,
                                                 shuffle=True, stratify=requiredTarget)
clf = svm.SVC() # Linear Kernel

#Train the model using the training sets
clf.fit(X_train, y_train)
pickle.dump(clf,open('model.pkl','wb'))