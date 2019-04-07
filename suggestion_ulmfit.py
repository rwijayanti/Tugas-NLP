from fastai import *
from fastai.text import *
from fastai.core import *
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet, stopwords
from nltk import pos_tag
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, precision_score, recall_score, f1_score
import re
import sys

lemmatizer = WordNetLemmatizer()
stopwords = set(stopwords.words('english'))

###Preprocessing
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
	text = stopwords_lemmatize_text(text)
	return text    

def define_model_ulmfit(X_train, y_train, X_test, y_test):
	df_train = pd.DataFrame(list(zip(X_train,y_train)), columns=['text','label'])
	df_test = pd.DataFrame(list(zip(X_test,y_test)), columns=['text','label'])

	data_lm = TextLMDataBunch.from_df(path = "", train_df = df_train, valid_df = df_test)
	learn = language_model_learner(data_lm, AWD_LSTM, drop_mult=0.5)
	learn.fit_one_cycle(1, 1e-2)
	
	#unfreeze pre-trained part and use “discriminative learning rate” 
	learn.unfreeze()
	learn.fit_one_cycle(1, 1e-3)
	#save the encoder
	learn.save_encoder('ft_enc')

	#classification
	data_clas = TextClasDataBunch.from_df(path = "", train_df = df_train, valid_df = df_test, vocab=data_lm.train_ds.vocab, bs=32)
	learn = text_classifier_learner(data_clas, AWD_LSTM, drop_mult=0.5)
	learn.load_encoder('ft_enc')
	print("Fitting Classifier Object")
	learn.fit_one_cycle(1, 1e-2)
	#learn.find.lr
	#learn.recorder.plot()
	
	'''
	print("Fitting Classifier Object after freezing all but last 2 layers")
	learn.freeze_to(-2)
	learn.fit_one_cycle(1, slice(5e-3/2., 5e-3))
	print("Fitting Classifier Object - discriminative learning")
	learn.unfreeze()
	learn.fit_one_cycle(1, slice(2e-3/100, 2e-3))
	'''
	return learn


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


def main(argv):
	input_train = argv[0]
	input_test = argv[1] 

	X_train, y_train = getData(input_train)
	X_test, y_test = getData(input_test)

	print ("Jumlah data training = ", len(X_train))
	print ("Jumlah data testing = ", len(X_test))
	

	#define model
	model = define_model_ulmfit(X_train, y_train, X_test, y_test)
	y_pred =[]
	for i in range(len(model.data.valid_ds)):
		__,cat = model.data.valid_ds[i]
		y_pred.append(int(str(cat)[-1])) 

	#evaluate model
	print("Precision: " ,precision_score(y_test,y_pred,average='weighted'))
	print("Recall: " ,recall_score(y_test,y_pred,average='weighted'))
	print("F1-score: " ,f1_score(y_test,y_pred,average='weighted'))
	print("Accuracy: " ,accuracy_score(y_test,y_pred,normalize=True))
	print(classification_report(y_test, y_pred, target_names=['suggestion','bukan suggestion']))

	
if __name__ == "__main__":
	#python suggestion_nn.py <data training> <data testing>
	main(sys.argv[1:])
