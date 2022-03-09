
import pandas as pd 
import os
os.chdir(r"/home/est_posgrado_enrique.santibanez/tesis/PolitiES/src")

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from test_evaluation import *

var_predictions = ["gender", "profession", "ideology_binary", "ideology_multiclass"]
reviews_politica = pd.read_csv("../data/raw/development.csv")
reviews_politica.head()

# Create a TFIDF Vectorizer using sci-kit. With this, we are going to represent all texts
# as counts of the vocabulary. 
vectorizer = TfidfVectorizer (
  analyzer = 'word',
  min_df = .1,
  max_features = 5000,
  lowercase = True
) 

# Get the TF-IDF values from the training set
X_train = vectorizer.fit_transform (reviews_politica['tweet'])

# Get the TF-IDF values from the test set
# Note that we apply the TF-IDF learned from the training split 
X_test = vectorizer.transform (reviews_politica['tweet'])

# We are going to store a baseline per trait
baselines = {}
# As we observed, this task is about four traits: two demographic and two psychographic. Therefore, we are going to
# train different and separate models for each task
for label in var_predictions:

# Get a baseline classifier
baselines[label] = SVC()


# Train the baseline for this label
baselines[label].fit(X_train, reviews_politica[label])
    
test_evaluation(X_test, reviews_politica[var_predictions], baselines,
               "/home/est_posgrado_enrique.santibanez/tesis/PolitiES/model/results/prueba.txt")
