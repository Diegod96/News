import numpy as np
from flask import Flask, request, render_template
from flask_cors import CORS
import os
import joblib
import pickle
import flask
import os
import newspaper
from newspaper import Article
import urllib



if __name__ == '__main__':
    with open('regression.pickle', 'rb') as handle:
        model = pickle.load(handle)

    # url = request.get_data(as_text=True)[5:]
    url = "https://abovethelaw.com/2020/07/sorry-to-interrupt-your-friday-but-homeland-security-is-disappearing-american-citizens-off-the-street/"
    url = urllib.parse.unquote(url)
    article = Article(str(url))
    article.download()
    article.parse()
    article.nlp()
    news = article.summary

    # Passing the news article to the model and returing whether it is Fake or Real
    pred = model.predict([news])
    print(pred)