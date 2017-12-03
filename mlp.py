import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.misc import imread
from wordcloud import WordCloud, STOPWORDS
from nltk.corpus import stopwords 

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.offline as py
import plotly.graph_objs as go
import plotly.tools as tls
from bs4 import BeautifulSoup

from sklearn import metrics

from sklearn.model_selection import cross_validate
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import learning_curve
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfTransformer,TfidfVectorizer, CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier

data = pd.read_csv('final.csv')

X_train, X_test, y_train, y_test= train_test_split(data.text, data.code, test_size=0.2, random_state=14)

np.random.seed(1)
scoring = {'acc': 'accuracy',
           'neg_log_loss': 'neg_log_loss',
           'f1_micro': 'f1_micro'}
kfolds = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
tfidf2 = CountVectorizer(ngram_range=(1, 1), 
                         stop_words='english',
                         lowercase = True, 
                         max_features = 5000)

<<<<<<< HEAD
clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(16, 16, 16), random_state=1)
=======
clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
>>>>>>> ed1ad2d848f601f09c18ba4d3c7a98f43c2ea612
model_clf = Pipeline([('tfidf2', tfidf2), ('clf', clf)])

results_clf = cross_validate(model_clf, X_train, y_train, cv=kfolds, scoring=scoring, n_jobs=-1)

print("CV Accuracy: {:0.4f} (+/- {:0.4f})".format(np.mean(results_clf['test_acc']),
                                                          np.std(results_clf['test_acc'])))
print("CV F1: {:0.4f} (+/- {:0.4f})".format(np.mean(results_clf['test_f1_micro']),
                                                          np.std(results_clf['test_f1_micro'])))
print("CV Logloss: {:0.4f} (+/- {:0.4f})".format(np.mean(-1*results_clf['test_neg_log_loss']),
                                                          np.std(-1*results_clf['test_neg_log_loss'])))

model_clf.fit(X_train, y_train)  
test_predicted_clf = model_clf.predict(X_test)

unique_type_list = ['INFJ', 'ENTP', 'INTP', 'INTJ', 'ENTJ', 'ENFJ', 'INFP', 'ENFP',
                    'ISFP', 'ISTP', 'ISFJ', 'ISTJ', 'ESTP', 'ESFP', 'ESTJ', 'ESFJ']

print(metrics.classification_report(y_test, test_predicted_clf, target_names=unique_type_list))

