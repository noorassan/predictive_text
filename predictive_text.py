from IPython import embed

import re
import pandas as pd

from random import sample
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from gensim.models import Word2Vec

def clean_text(text):
	text = text.lower()

	#removing punctuation except for ./!/?
	text = re.sub('[^\w\s.!?]', '', text)

	return(text)

data = open("book.txt").read()
paragraph_list = data.split('\n\n')

#remove line breaks
paragraph_list = [paragraph.replace('\n', ' ') for paragraph in paragraph_list]

#lowercase and strip punctuation except for ./!/?
paragraph_list = [clean_text(paragraph) for paragraph in paragraph_list]

#insert <stp> in place of ./!/?
stp_paragraph_list = []
for paragraph in paragraph_list:
	stp_paragraph = ''
	for word in paragraph.split():
		split_word = re.split(r'[.!?]', word)
		if(split_word[-1] == '' and split_word[-2] not in ["dr", "mr", "mrs", "ms"]):
			stp_paragraph += split_word[-2]
			stp_paragraph += " <stp>"
		else:
			stp_paragraph += word
		
		stp_paragraph += ' '
	
	stp_paragraph_list.append(stp_paragraph)

#remove empty list items
stp_paragraph_list = [paragraph for paragraph in stp_paragraph_list if len(paragraph) > 0]

#setup df
df = pd.DataFrame(columns=['input', 'output'])

#populate df
for paragraph in stp_paragraph_list:
	word_list = paragraph.split()
	n = 0
	while n < len(word_list) - 3:
		in_words = " ".join(word_list[n:n+3])
		out_word = word_list[n+3]
		n += 1	

		df = df.append({'input': in_words, 'output': out_word}, ignore_index=True)

#sample
df = df.sample(n=5000)

#change paragraphs into lists of words for word2vec
stp_paragraph_list = [[word for word in paragraph.split()] for paragraph in stp_paragraph_list]

#vectorize
word2vec = Word2Vec(stp_paragraph_list)
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
