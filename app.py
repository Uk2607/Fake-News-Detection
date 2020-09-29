import warnings

warnings.filterwarnings('ignore')
from tensorflow.keras.models import load_model
import nltk
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle
import os
from flask import Flask, request, render_template, url_for, redirect, jsonify
from flask_cors import CORS, cross_origin


app = Flask(__name__)
CORS(app)


nltk.download('stopwords')
nltk.download('punkt')


def loadModels(model_path, encoder_path):
    model_path = os.path.join(model_path,"model.h5")
    encoder_path = os.path.join(encoder_path,"tokenizer.tk")
    model = load_model(model_path)
    with open(encoder_path, 'rb') as pickle_file:
        encoder = pickle.load(pickle_file)
    return model, encoder


def preprocessing(par):
    stop_words = set(nltk.corpus.stopwords.words("english"))
    tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
    tmp = []
    sentences = nltk.sent_tokenize(par)
    for sent in sentences:
        sent = sent.lower()
        tokens = tokenizer.tokenize(sent)
        filtered_words = [w.strip() for w in tokens if w not in stop_words and len(w) > 1]
        tmp.extend(filtered_words)
    return tmp


def transform(X, maxlen, encoder):
    tmp = np.array(X)
    tmp = tmp.reshape(1, tmp.shape[0])
    X = encoder.texts_to_sequences(tmp.tolist())
    return pad_sequences(X, maxlen)


def predict_news(txt, maxlen, clf_model, clf_encoder):
    X = preprocessing(txt)
    X = transform(X, maxlen, clf_encoder)
    y = clf_model.predict(X)
    if y > 0.5:
        return "Real"
    else:
        return "Fake"


@app.route('/')
def home():
    return render_template("index.html")
    
    
@app.route('/covid', methods=['GET', 'POST'])
def covid_func():
    if request.method == 'POST':
        return redirect(url_for('home'))
    return render_template("covid.html")
    

@app.route("/predict", methods=['POST'])
def predict():
    model, encoder = loadModels('models', 'models')
    #model, encoder = loadModels('E:\\New folder\\Fake-Local\\models', 'E:\\New folder\\Fake-Local\\models')
    req = request.form
    news = req.get("searchtxt")
    prediction = predict_news(str(news), 256, model, encoder)
    if prediction:
        response = {'btnMessage': 'News is {}'.format(prediction)}
    else:
        response = {'btnMessage': 'News is {}'.format(prediction)}
    
    return jsonify(response), 200


if __name__ == "__main__":
    app.run(debug=True)
