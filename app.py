import os
import pickle
import urllib
import flask
from flask import Flask, request, render_template
from flask_cors import CORS
from newspaper import Article

app = Flask(__name__)
CORS(app)
app = flask.Flask(__name__, template_folder='templates')

# Load the models
with open('naive.pickle', 'rb') as handle:
    naive = pickle.load(handle)

with open('sgd.pickle', 'rb') as handle:
    sgd = pickle.load(handle)

with open('regression.pickle', 'rb') as handle:
    regression = pickle.load(handle)


@app.route('/')
def main():
    return render_template('main.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """
    Handles prediction portion of the application
    Gets the news via a url and extracts the text
    Runs the text through the three models
    If a majority of the models deem the article to be FAKE then FAKE will be returned
    Else TRUE will be returned
    :return: render_template
    """

    # Get the news article url
    url = request.get_data(as_text=True)[5:]
    url = urllib.parse.unquote(url)
    article = Article(str(url))

    # Download the article
    # Parse the text into "news" variable
    article.download()
    article.parse()
    article.nlp()
    news = article.summary

    # Setup counters for the number of FAKE and TRUE results from the models
    FAKE = 0
    TRUE = 0

    # Run the news through the Logistic Regression model
    pred = regression.predict([news])
    regression_prediction = pred[0]
    if regression_prediction == 'FAKE':
        FAKE += 1
    else:
        TRUE += 1

    # Run the news through the Multinominal Naive Bayes model
    pred = naive.predict([news])
    naive_prediction = pred[0]
    if naive_prediction == 'FAKE':
        FAKE += 1
    else:
        TRUE += 1

    # Run the news through the SGD Classification model
    pred = sgd.predict([news])
    sgd_prediction = pred[0]
    if sgd_prediction == 'FAKE':
        FAKE += 1
    else:
        TRUE += 1

    # Assign "pred" to either FAKE or TRUE based on the results
    if FAKE > TRUE:
        pred = 'FAKE'
    else:
        pred = "TRUE"

    # Return the result to the viewer
    return render_template('main.html',
                           regression_text=f'Logistic Regression determined the news article to be {regression_prediction}',
                           naive_text=f'Multinominal Naive Bayes determined the news article to be {naive_prediction}',
                           sgd_text=f'SGD Classification determined the news article to be {sgd_prediction}',
                           prediction_text=f'Overall, this news articles as been determined to be {pred}')


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(port=port, debug=True, use_reloader=False)
