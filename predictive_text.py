import re
from itertools import chain

import pandas as pd

from nltk.tokenize import sent_tokenize, word_tokenize

from gensim.models import Word2Vec
import gensim.downloader as api

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def clean_text(text):
    text = text.lower()

    #removing punctuation except for ./!/?
    text = re.sub('[^\w\s.!?]', '', text)

    return(text)

def retrieve_vectors(words, model):
    vectors = []
    for word in words:
        try:
            vectors.append(model[word])
        except KeyError:
            return

    return vectors

text = open("book.txt")
text = text.read()
text = clean_text(text)

#convert data into nested list of sentences which are lists of words
data = [word_tokenize(sentence) for sentence in sent_tokenize(text)]

#setup df
df = pd.DataFrame(columns=['input', 'vectors', 'output'])

#load vectors
model = api.load("glove-twitter-25")

#populate df
for sentence in data:
    n = 0
    while n < len(sentence) - 3:
        in_words = sentence[n:n+3]
        out_word = sentence[n+3]
        n += 1

        vectors = retrieve_vectors(in_words, model)

        if vectors:
            #conglomerate vectors into single list
            vectors = [vector.tolist() for vector in vectors]
            vectors = list(chain.from_iterable(vectors))        

        
            df = df.append({'input': in_words, 'vectors': vectors,  'output': out_word}, ignore_index=True)

#sample data
df = df.sample(n=2000)

#split data into train and test - including 'input' in x so it can be connected w/ output
x_train, x_test, y_train, y_test = train_test_split(df[['input', 'vectors']], df['output'], test_size = 0.2)

#random forest classifier

rf = RandomForestClassifier(n_estimators = 100)
rf.fit(pd.DataFrame(x_train["vectors"].tolist()), y_train)

predictions = rf.predict(pd.DataFrame(x_test["vectors"].tolist()))

#compare prediction and actual
matches = 0
for in_words, prediction, actual in zip(x_test['input'], predictions, y_test):
    print(in_words, "\t\tprediction:", prediction, "\t\tactual:", actual)
    if prediction == actual:
        matches += 1

print("matches:", matches)
print(len(predictions))
