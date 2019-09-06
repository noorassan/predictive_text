import re
import pandas as pd
import gensim.downloader as api
import numpy as np


from itertools import chain
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from gensim.models import Word2Vec


def clean_text(text):
    text = text.lower()

    # removing punctuation except for ./!/?
    text = re.sub('[^\w\s.!?]', '', text)

    return(text)

def retrieve_vectors(words, model):
    vectors = []
    for word in words:
        try:
            vectors.append(model[word])
        except KeyError:
            return

		# conglomerate vectors into single one-dimensional list
    vectors = [vector.tolist() for vector in vectors]
    vectors = list(chain.from_iterable(vectors))        

    return vectors

def next_word(in_words, rf, model):
		vectors = retrieve_vectors(in_words, model)
		vectors = np.array(vectors)
		
		return rf.predict(vectors.reshape(1, -1))[0]
		

text = open("book.txt")
text = text.read()
text = clean_text(text)

# convert data into nested list of sentences which are lists of words
data = [word_tokenize(sentence) for sentence in sent_tokenize(text)]

# setup df
df = pd.DataFrame(columns=['input', 'output'])

# load vectors
model = api.load("glove-twitter-25")

# populate df and vectors list
vectors_list = []
for sentence in data:
    n = 0
    while n < len(sentence) - 3:
        in_words = sentence[n:n+3]
        out_word = sentence[n+3]
        n += 1

        vectors = retrieve_vectors(in_words, model)

        if vectors:
            vectors_list.append(vectors)	
            df = df.append({'input': in_words, 'output': out_word}, ignore_index=True)

#add vectors to df
vectors_df = pd.DataFrame(vectors_list)
vectors_df.index = df.index
df = pd.concat([df, vectors_df], axis=1)

df = df.sample(n=2000)

# split data into train and test - including 'input' in x so it can be connected w/ output
columns = [c for c in df.columns if c != 'output']

x_train, x_test, y_train, y_test = train_test_split(df[columns], df['output'], test_size = 0.2)

# random forest classifier
columns = [c for c in columns if c != 'input']

rf = RandomForestClassifier(n_estimators = 100)
rf.fit(x_train[columns], y_train)

predictions = rf.predict(x_test[columns])

# compare prediction and actual
matches = 0
for in_words, prediction, actual in zip(x_test['input'], predictions, y_test):
    print(in_words, "\t\tprediction:", prediction, "\t\tactual:", actual)
    if prediction == actual:
    		matches += 1

print("matches:", matches)
print(len(predictions))

# generate text
seeds = [["then", "i", "went"], ["i", "told", "mr."], ["did", "you", "ever"]]

predictions = [next_word(seed, rf, model) for seed in seeds]

for in_words, prediction in zip(seeds, predictions):
		print(" ".join(in_words), prediction)
