import os, sys, random
import pickle
import csv, sklearn, matplotlib

# Importing Count Vectorizer and Word Vector
file_wd = os.path.dirname(__file__)
word_vec_path = "../models/word_vec.sav"
tfidf_path = "../models/tfidf.sav"
word_vec = pickle.load(open(os.path.join(file_wd, word_vec_path), "rb"))
tfidf = pickle.load(open(os.path.join(file_wd, tfidf_path), "rb"))

# Defining Score Function
def score(text: str, model, threshold: float, word_vec = word_vec, tfidf = tfidf): 
	text_trans = word_vec.transform([text])
	text_tfidf = tfidf.transform(text_trans)[0]
	propensity = model.predict_proba(text_tfidf)[0][1]
	if propensity >= threshold:
		prediction = bool(1)
	else:
		prediction = bool(0)

	return prediction, propensity