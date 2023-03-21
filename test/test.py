import os, sys, random, time
import pickle, requests
import numpy as np
import warnings
warnings.simplefilter("ignore", category=UserWarning) 

# Local Imports
sys.path.insert(1, os.path.join(os.path.dirname(__file__), "../test"))

from score import *

# Importing Model
file_wd = os.path.dirname(__file__)
lr_model_path = "../models/lr_model.sav"
lr_model = pickle.load(open(os.path.join(file_wd, lr_model_path), "rb"))

# Defining input values to test the score function on
obvious_ham = "Where are you?"
obvious_spam = "Press this link to win an aeroplane for free"
threshold = 0.5

# Defining Unit Tests

# Smoke Test: Function returns values without crashing
def test_smoke(text=obvious_ham, threshold=threshold, model=lr_model) -> None:
    label, prop = score(text, model, threshold)

    assert label != None, "Prediction is not boolean"
    assert prop != None, "Propensity is not float"

# Format Test: Check function input/output types 
def test_input_formats(text=obvious_ham, threshold=threshold, model=lr_model) -> None:
    label, prop = score(text, model, threshold)

    assert type(text) == str
    assert type(threshold) == float 
    assert type(label) == bool
    assert type(prop) == np.float64 

# Prediction Value Test
def test_pred_value(text=obvious_ham, threshold=threshold, model=lr_model) -> None:
    label, prop = score(text, model, threshold)

    assert label == False or label == True

# Propensity Value Test
def test_prop_value(text=obvious_ham, threshold=threshold, model=lr_model) -> None:
    label, prop = score(text, model, threshold)

    assert prop >= 0 and prop <= 1

# Check Prediction Value for Threshold = 0 Test
def test_pred_thres_zero(text=obvious_ham, threshold=threshold, model=lr_model) -> None:
    label, prop = score(text, model, threshold=0)

    assert label == True

# Check Prediction Value for Threshold = 1 Test
def test_pred_thres_one(text=obvious_ham, threshold=threshold, model=lr_model) -> None:
    label, prop = score(text, model, threshold=1)

    assert label == False

# Check Prediction Value for Threshold = 1 Test
def test_obvious_spam(text=obvious_spam, threshold=threshold, model=lr_model) -> None:
    label, prop = score(text, model, threshold)

    assert label == True, "Prob: {}".format(prop)

# Check Prediction Value for Threshold = 1 Test
def test_obvious_ham(text=obvious_ham, threshold=threshold, model=lr_model) -> None:
    label, prop = score(text, model, threshold)

    assert label == False

def test_flask():
        # Launch the Flask app using os.system
        os.system('python src/app.py &')

        # Wait for the app to start up
        time.sleep(1)

        # Make a request to the endpoint
        response = requests.get('http://127.0.0.1:5000/')
        print(response.status_code)

        # Assert that the response is what we expect
        assert response.status_code == 200

        assert type(response.text) == str

        # Shut down the Flask app using os.system
        os.system('kill $(lsof -t -i:5000)')