from collections import defaultdict
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, precision_score, recall_score, f1_score
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
import fasttext as ft
import numpy as np
import pandas as pd
import re
import pickle
import sys
import csv

lemmatizer = WordNetLemmatizer()
#stopwords = set(stopwords.words('english'))

###Preprocessing
'''
def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

def stopwords_lemmatize_text(text):
    word_tokens = word_tokenize(text)
    text_st = [w for w in word_tokens if not w in stopwords]
    text = ' '.join([lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in text_st])
    return text
'''
def lemmatize_text(text):
    word_tokens = word_tokenize(text)
    text = ' '.join([lemmatizer.lemmatize(w,"a") for w in word_tokens])
    return text

def processText(text):
     #Convert to lower case
	text = text.lower()
	#Remove url
	text = re.sub('((www\.[^\s]+)|(https?:[^\s]+))',' ',text)
	#Remove word not starting with letters
	text = re.sub("[^a-zA-Z\s][a-zA-Z0-9]*", '',text)
	#Remove ____ character
	text = re.sub(r'____', r' ', text)
	#Remove punctuation
	text = re.sub('[\?,.!(){}]+',' ',text)
	#trim
	text = text.strip()
	#lemmatize
	text = lemmatize_text(text)
	return text    
	
def getData(filename):
	data = pd.read_csv(filename, delimiter=',',names=['id','text','label'],engine='python')

	X = []
	y = []
	wordsFiltered = []
	for i in range(len(data)):
		text = data["text"][i]
		label = data["label"][i]
		processedText = processText(text)
		X.append(processedText)
		y.append(label)

	return X,y

class getTag():
	def fit(self, x, y=None):
		return self

	def transform(self, posts):
		X_tag = []
		for sent in posts:
			tagged_sent = pos_tag(word_tokenize(sent))
			tags = [i[1] for i in tagged_sent]
			pos_match = any(elem in ['MD', 'VB'] for elem in tags)
			X_tag.append(pos_match)
		X_tag = np.array([X_tag]).T
		return X_tag

class getSuggestionWord():
	def fit(self, x, y=None):
		return self

	def transform(self, posts):
		keywords = ["suggest","recommend","hope","hopefully","go","request","wish","want","add","provide","fix","make","should", "would","could", "can","need","allow","ask", "intend", "do","please", "pls"]
		X_suggestion = []
		for sent in posts:
			tokenized_sent = word_tokenize(sent)
			keyword_match = any(elem in keywords for elem in tokenized_sent)
			X_suggestion.append(keyword_match)
		X_suggestion = np.array([X_suggestion]).T
		return X_suggestion   

def classifier(X,y, algorithm):
	if algorithm=="svm":
		alg = LinearSVC() 
	elif algorithm=="lr":
		alg = LogisticRegression()
	elif algorithm=="dt":
		alg = DecisionTreeClassifier()	
	
	model =  Pipeline([
			('features', FeatureUnion([
				('feat1',TfidfVectorizer()),
				('feat2',CountVectorizer(ngram_range=(1,3))),
				('feat3',getSuggestionWord()),
				('feat4',getTag())
			])),
			('mdl',alg)
		])  

	model.fit(X,y)
	return model


def main(argv):
	input_train = argv[0]
	input_test = argv[1] 
	X_train, y_train = getData(input_train)
	X_test, y_test = getData(input_test)
	print ("Jumlah data training = ", len(X_train))
	print ("Jumlah data testing = ", len(X_test))
	vec_clf_svm = classifier(X_train,y_train, "svm")
	vec_clf_lr = classifier(X_train,y_train, "lr")
	vec_clf_dt = classifier(X_train,y_train, "dt")
	y_pred_svm = vec_clf_svm.predict(X_test)
	y_pred_lr = vec_clf_lr.predict(X_test)
	y_pred_dt = vec_clf_dt.predict(X_test)
	print ("Hasil evaluasi SVM" )
	print("Precision: " ,precision_score(y_test,y_pred_svm,average='weighted'))
	print("Recall: " ,recall_score(y_test,y_pred_svm,average='weighted'))
	print("F1-score: " ,f1_score(y_test,y_pred_svm,average='weighted'))
	print("Accuracy: " ,accuracy_score(y_test,y_pred_svm,normalize=True))
	print(classification_report(y_test, y_pred_svm, target_names=['suggestion','bukan suggestion']))

	print ("Hasil evaluasi Logistic Regression" )
	print("Precision: " ,precision_score(y_test,y_pred_lr,average='weighted'))
	print("Recall: " ,recall_score(y_test,y_pred_lr,average='weighted'))
	print("F1-score: " ,f1_score(y_test,y_pred_lr,average='weighted'))
	print("Accuracy: " ,accuracy_score(y_test,y_pred_lr,normalize=True))
	print(classification_report(y_test, y_pred_lr, target_names=['suggestion','bukan suggestion']))

	print ("Hasil evaluasi Decision Tree" )
	print("Precision: " ,precision_score(y_test,y_pred_dt,average='weighted'))
	print("Recall: " ,recall_score(y_test,y_pred_dt,average='weighted'))
	print("F1-score: " ,f1_score(y_test,y_pred_dt,average='weighted'))
	print("Accuracy: " ,accuracy_score(y_test,y_pred_dt,normalize=True))
	print(classification_report(y_test, y_pred_dt, target_names=['suggestion','bukan suggestion']))

if __name__ == "__main__":
	#python suggestion_ml.py <data training> <data testing>
	main(sys.argv[1:])


	