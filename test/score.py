import os, sys, random
import pickle
import csv, nltk, sklearn, matplotlib
import re, contractions

from collections import Counter
from nltk.tokenize import RegexpTokenizer, word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer

# Importing Count Vectorizer and Word Vector
file_wd = os.path.dirname(__file__)
word_vec_path = "../models/word_vec.sav"
tfidf_path = "../models/tfidf.sav"
word_vec = pickle.load(open(os.path.join(file_wd, word_vec_path), "rb"))
tfidf = pickle.load(open(os.path.join(file_wd, tfidf_path), "rb"))

def get_words(sent):
	sent = re.sub(r'^https?:\/\/.*[\r\n]*', '', sent, flags=re.MULTILINE) # Remove urls starting with http
	sent = re.sub(r'^http?:\/\/.*[\r\n]*', '', sent, flags=re.MULTILINE) # Remove urls starting with https
	sent = contractions.fix(sent, slang=True) # Replace contractions with words
	sent = ''.join([i for i in sent if not i.isdigit()]) # Remove numbers
	tokenizer = RegexpTokenizer(r'\w+')
	tokenized_words = tokenizer.tokenize(sent) # Remove all punctuation marks (don't have to worry about contractions)
	return tokenized_words

def get_tokenized_sms(sms):
	words_list = get_words(sms)
	
	stopwords = list(nltk.corpus.stopwords.words('english'))
	lemmatizer = WordNetLemmatizer()
	temp_list = []

	for word in words_list:
		if len(word) > 1 and word.lower() not in stopwords:
			word = lemmatizer.lemmatize(word.lower())
			temp_list.append(word.lower())

	return temp_list

def get_string_from_list_words(word_list):
	temp = ""
	for i in word_list:
		temp = temp + i + " "

	return temp

# Defining Score Function
def score(text: str, model, threshold: float, word_vec = word_vec, tfidf = tfidf): 
	word_list = get_tokenized_sms(text)
	regen_text = get_string_from_list_words(word_list)
	
	text_trans = word_vec.transform([text])
	text_tfidf = tfidf.transform(text_trans)[0]
	propensity = model.predict_proba(text_tfidf)[0][1]
	if propensity >= threshold:
		prediction = bool(1)
	else:
		prediction = bool(0)

	return prediction, propensity