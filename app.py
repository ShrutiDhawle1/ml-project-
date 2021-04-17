import streamlit as st
import pandas as pd
import numpy as np
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import nltk
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV

stop_words = set(stopwords.words('english'))
df1 = pd.read_csv('amazon_alexa.tsv',sep='\t',usecols=['rating','verified_reviews'])
df1.columns = ['rating','review']
df2 = pd.read_csv('flipkart_reviews_dataset.csv',usecols=['rating','review'])
frames = [df1,df2]
df = pd.concat(frames)

#Preprocessing
def preprocessing(x):
  x = x.lower()
  x = re.sub(r"http\S+|www\S+", "",x,flags=re.MULTILINE)
  x = x.translate(str.maketrans("", "", string.punctuation))

  x_token = word_tokenize(x)
  filtered_words = [word for word in x_token if word not in stop_words]
  return " ".join(filtered_words)

def remove_numbers(text):
    result = re.sub(r'\d+', ' ', text)
    return result

df.review = df.review.apply(preprocessing)
df.review = df.review.apply(remove_numbers)

tfidf = TfidfVectorizer(max_features=20000, ngram_range=(1,5), analyzer='char',lowercase=True)
X = tfidf.fit_transform(df.review)
y = df['rating']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3, random_state = 0)
clf = LinearSVC(C=30)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

st.title("Amazon\Flipkart Rating Prediction\n",)
st.text("Please Clear Cache Before The Run")
st.image('pngaaa.com-4102078.png', caption=None, width=400, use_column_width=None, clamp=False, channels='RGB', output_format='auto')
user_input = st.text_input("Input Any Reviews : ")

l = tfidf.transform([user_input])
t = clf.predict(l)
t = int(t)
st.write(f'Ratings : ',t,'Star')