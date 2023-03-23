import os, sys, random
import pickle

from flask import Flask, request, render_template

# Local Imports
sys.path.append(os.path.join(os.path.dirname(__file__), "../test"))

from score import score

app = Flask(__name__, template_folder = '../test/template')

# Importing Model
file_wd = os.path.dirname(__file__)
lr_model_path = "../models/lr_model.sav"
lr_model = pickle.load(open(os.path.join(file_wd, lr_model_path), "rb"))

# Setting threshold value
threshold=0.5

@app.route('/') 
def home():
    return render_template('spam.html')

@app.route('/spam', methods=['POST'])
def spam():
    sent = request.form['sent']
    label, prop = score(sent, lr_model, threshold)
    prop = round(prop, 3)
    lbl = "Spam" if label == 1 else "Not spam"
    ans1 = f"""The input text: {sent}"""
    ans2 = f"""The prediction: {lbl}""" 
    ans3 = f"""The propensity score: {prop}"""
    return render_template('result.html', ans1 = ans1, ans2 = ans2, ans3 = ans3)

if __name__ == '__main__': 
    app.run(host="0.0.0.0", port=5000, debug=True)