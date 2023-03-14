import score
import pickle
import numpy as np
import os
import requests
import time
import unittest
from multiprocessing import Process
from app import app

filename = open("test",'rb')
mlp =pickle.load(filename)

sent="I like dogs"
threshold=0.4
label,prop=score.score(sent,mlp,threshold)

class TestFunction:

    # check if score function returns values properly
    def smoke_test(self):
        assert label!= None
        assert prop!= None
            
    # check if the type of data meets certain requirements
    def format_test(self):
        assert type(sent) == str
        assert type(threshold) == float 
        assert type(label) == np.int64
        assert type(prop) == np.float64 

    # check if the label value is in {True,False}
    def pred_value(self):
        assert label == 0 or label == 1

    # check if propensity lies in [0,1]
    def propensity_value(self):
        assert prop>=0 and prop<=1

    # if threshold is 0, prediction becomes True
    def prop_test_0(self):
        label,prop=score.score(sent,mlp,0)
        assert label == 1

    # if threshold is 1, prediction becomes False
    def prop_test_1(self):
        label,prop=score.score(sent,mlp,1)
        assert label == 0

    # testing obvious spam
    def test_spam(self):
        label,prop=score.score("You have won a jackpot. Get 5 million dollars, follow steps.",mlp,threshold)
        assert label == 1

    # testing obvious ham
    def test_ham(self):
        label,prop=score.score("Cats are cute.",mlp,threshold)
        assert label == 0




class TestFlask(unittest.TestCase):
    
    def test_flask(self):
        # Launch the Flask app using os.system
        os.system('python app.py &')

        # Wait for the app to start up
        time.sleep(1)

        # Make a request to the endpoint
        response = requests.get('http://127.0.0.1:5000/')
        print(response.status_code)

        # Assert that the response is what we expect
        self.assertEqual(response.status_code, 200)
        # print("OK")
        self.assertEqual(type(response.text), str)
        # print("OKAY")

        # Shut down the Flask app using os.system
        os.system('kill $(lsof -t -i:5000)')


if __name__ == '__main__':
    unittest.main()

