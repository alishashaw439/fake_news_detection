import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report,confusion_matrix
import pickle

news=pd.read_csv('/home/alisha/Documents/fakeNews_website/news.csv')
X=news['text']
y=news['label']
news.head()



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

pipeline=Pipeline([('tfidf', TfidfVectorizer(stop_words='english')),
                   ('nbmodel', MultinomialNB())])



pipeline.fit(X_train, y_train)

pred=pipeline.predict(X_test)
print(pred)
print(classification_report(y_test,pred))
print(confusion_matrix(y_test,pred))



with open('model.pickle', 'wb') as handle:
    pickle.dump(pipeline, handle, protocol=pickle.HIGHEST_PROTOCOL)

