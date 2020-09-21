import numpy as np
import pandas as pd
import itertools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import pickle

df=pd.read_csv('/home/alisha/Documents/fakeNews_website/news.csv')
df.shape
df.head()

labels=df.label
labels.head()

x_train, x_test, y_train, y_test = train_test_split(df['text'],labels, test_size=0.2, random_state=7)

tfidf_vectorizer=TfidfVectorizer(stop_words='english',max_df=0.7)
tfidf_train=tfidf_vectorizer.fit_transform(x_train)
tfidf_test=tfidf_vectorizer.transform(x_test)

pac=PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train, y_train)

y_pred=pac.predict(tfidf_test)
score=accuracy_score(y_test, y_pred)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

#Fitting model with trainig data
regressor.fit(x                           , y)

# Saving model to disk
pickle.dump(regressor, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(f'Accuracy: {round(score*100,2)}%')


confusion_matrix(y_test, y_pred, labels=['FAKE','REAL'])
