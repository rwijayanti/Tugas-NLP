from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Embedding, LSTM, Bidirectional, RepeatVector, TimeDistributed
from keras.preprocessing import text, sequence
from keras.utils import to_categorical, vis_utils 
from nltk.corpus import wordnet
from keras_self_attention import SeqSelfAttention
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from nltk import pos_tag
from tqdm import tqdm
import numpy as np
import pandas as pd
import re
import sys
import codecs

lemmatizer = WordNetLemmatizer()
plt.style.use('ggplot')

###Preprocessing
def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

def lemmatize_text(text):
    text = word_tokenize(text)
    text = ' '.join([lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in text])
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

def create_tokenizer(lines):
	tokenizer = text.Tokenizer()
	tokenizer.fit_on_texts(lines)
	return tokenizer

def encode_docs(tokenizer, max_length,docs,label):
	encoded = tokenizer.texts_to_sequences(docs)
	X_padded = sequence.pad_sequences(encoded,maxlen=max_length, padding='post')
	#y_cat = to_categorical(label)
	y_cat = np.array([label]).T
	return X_padded, y_cat


def define_model_bilstm(vocab_size, max_length, num_classes):
	model = Sequential()
	model.add(Embedding(vocab_size, 50, input_length=max_length))
	model.add(Bidirectional(LSTM(64)))
	model.add(Dense(10, activation='relu'))
	model.add(Dense(num_classes, activation='sigmoid'))
	model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
	model.summary()
	return model

def define_model_bilstm_pretrained(vocab_size, max_length, num_classes, embedding_weight,st_train):
	model = Sequential()
	model.add(Embedding(vocab_size, 50, input_length=max_length, weights=[embedding_weight], trainable=st_train))
	model.add(Bidirectional(LSTM(64)))
	model.add(Dense(10, activation='relu'))
	model.add(Dense(num_classes, activation='sigmoid'))
	model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
	model.summary()
	return model

def plot_history(history):
	acc = history.history['acc']
	val_acc = history.history['val_acc']
	loss = history.history['loss']
	val_loss = history.history['val_loss']

	plt.figure(figsize=(12, 5))
	plt.subplot(1, 2, 1)
	plt.plot(acc)
	plt.plot(val_acc)
	plt.title('Model accuracy')
	plt.ylabel('accuracy')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
	plt.subplot(1, 2, 2)
	plt.plot(loss)
	plt.plot(val_loss)
	plt.title('Model loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
	plt.show()


def getEmbeddingGlove():
	em_file = 'glove/glove.twitter.27B.50d.txt'
	 
	embeddings = {}
	with open(em_file) as f:
		for line in f:
			values = line.split()
			word = values[0]
			coefs = np.asarray(values[1:], dtype='float32')
			embeddings[word] = coefs
	return embeddings

def getEmbedding(tokenizer, vocab_size):
	vector_length = 50
	embeddings_index = np.zeros((vocab_size, vector_length))
	word_index = tokenizer.word_index
	vector = getEmbeddingGlove()
	
	for word, idx in word_index.items():
		try:
			embeddings_index[idx] = vector[word]
		except:
			pass
	return embeddings_index

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

def getData_train(filename):
	X_train,y_train = getData(filename)
	tokenizer = create_tokenizer(X_train)
	vocab_size = len(tokenizer.word_index) + 1
	print('Vocabulary size: %d' % vocab_size)
	max_length = max([len(s.split()) for s in X_train])
	print('Maximum length: %d' % max_length)
	return tokenizer, vocab_size, max_length, X_train, y_train

def train_test_model(model, X_train, y_train, X_test, y_test, n_epochs):
	history = model.fit(X_train, y_train, epochs=n_epochs, verbose=2, validation_data=(X_test, y_test), batch_size=32)
	y_pred = model.predict_classes(X_test, verbose=0)
	y_pred = y_pred[:,0]
	#evaluate model
	print("Precision: " ,precision_score(y_test,y_pred,average='weighted'))
	print("Recall: " ,recall_score(y_test,y_pred,average='weighted'))
	print("F1-score: " ,f1_score(y_test,y_pred,average='weighted'))
	print("Accuracy: " ,accuracy_score(y_test,y_pred,normalize=True))
	print(classification_report(y_test, y_pred, target_names=['suggestion','bukan suggestion']))
	return history
	

def main(argv):
	input_train = argv[0]
	input_test = argv[1] 
	num_classes = 1
	num_epochs = 5
	st_train = True

	tokenizer, vocab_size, max_length,X_train, y_train = getData_train(input_train)
	X_train, y_train = encode_docs(tokenizer, max_length, X_train, y_train)
	
	X_test, y_test = getData(input_test)
	X_test, y_test = encode_docs(tokenizer, max_length, X_test, y_test)

	print ("Jumlah data training = ", len(X_train))
	print ("Jumlah data testing = ", len(X_test))
	
	embedding_weight = getEmbedding(tokenizer, vocab_size)

	results = pd.DataFrame()
	
	#define model
	model = define_model_bilstm(vocab_size, max_length, num_classes)
	history = train_test_model(model,X_train, y_train, X_test, y_test, num_epochs)
	vis_utils.plot_model(model,to_file='model_bilstm.png',show_shapes=True, show_layer_names=True)
	plot_history(history)
	
	model = define_model_bilstm_pretrained(vocab_size, max_length, num_classes, embedding_weight, True)
	history = train_test_model(model,X_train, y_train, X_test, y_test, num_epochs)
	vis_utils.plot_model(model,to_file='model_pretrained_glove.png',show_shapes=True, show_layer_names=True)
	plot_history(history)


if __name__ == "__main__":
	#python suggestion_nn.py <data training> <data testing>
	main(sys.argv[1:])
