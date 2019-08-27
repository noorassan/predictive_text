from IPython import embed

import re
import pandas as pd

from random import sample
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from gensim.models import Word2Vec
from nltk.tokenize import sent_tokenize, word_tokenize

def clean_text(text):
    text = text.lower()

    #removing punctuation except for ./!/?
    text = re.sub('[^\w\s.!?]', '', text)

    return(text)


text = open("book.txt")
text = text.read()
text = clean_text(text)

#convert data into nested list of sentences which are lists of words
data = [word_tokenize(sentence) for  sentence in sent_tokenize(text)]

#setup df
df = pd.DataFrame(columns=['input', 'output'])

#populate df
for sentence in data:
   n = 0
   while n < len(sentence) - 3:
	in_words = " ".join(sentence[n:n+3])
	out_word = sentence[n+3]
	n += 1	

	df = df.append({'input': in_words, 'output': out_word}, ignore_index=True)

#sample
df = df.sample(n=5000)

#vectorize
word2vec = Word2Vec(data)
embed()


#split data into train and test - including 'input' in x so it can be connected w/ output
columns = [c for c in df.columns if c != 'output']

x_train, x_test, y_train, y_test = train_test_split(df[columns], df['output'], test_size = 0.2)

#random forest classifier
columns = [c for c in x_train.columns if c != 'input']

rf = RandomForestClassifier(n_estimators = 100)
rf.fit(x_train[columns], y_train)

predictions = rf.predict(x_test[columns])

#compare prediction and actual
matches = 0
for in_words, prediction, actual in zip(x_test['input'], predictions, y_test):
    print(in_words, "\t\tprediction:", prediction, "\t\tactual:", actual)
    if prediction == actual:
        matches += 1

print("matches:", matches)
print(len(predictions))
