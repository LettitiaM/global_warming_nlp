from crypt import methods
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from flask import Flask, render_template, request
import pickle
from sklearn.neighbors import KNeighborsClassifier
import tweet_preprocessor
import model_script

string = 'linearSVC'
linearSVC = model_script.loadModel(string)

app = Flask(__name__,template_folder='template')
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/result', methods=['POST'])
def result():
    if request.method == 'POST':
        tweet = format(request.form['tweet'])
        input = tweet_preprocessor.preprocess_tweet(tweet)
        input = [input]

        y_pred = model_script.getPrediction(linearSVC,input)
        prediction = y_pred
    
        return render_template("result.html", prediction = prediction)
    
1
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5534, debug=True)
