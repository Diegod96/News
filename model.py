import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
import pickle

from sklearn.tree import DecisionTreeClassifier


def clean_data(file_paths):
    """
    Clean the two data sets
    Combine into one main news dataset
    :param file_paths:
    :return: df
    """

    # Get the datasets
    fake = pd.read_csv(file_paths[0])
    real = pd.read_csv(file_paths[1])

    # Assign labels to the datasets
    fake['label'] = 'FAKE'
    real['label'] = 'TRUE'

    # Merge both datasets
    frames = [fake, real]
    df = pd.concat(frames)

    # Delete any "http" strings
    patternDel = "http"
    filter1 = df['date'].str.contains(patternDel)

    # Apply filter to the dataframe
    df = df[~filter1]

    # Sort values by date
    df = df.sort_values(by=['date'])
    df = df.reset_index(drop=True)

    # Combine the "subject", "title", and "text" columns into one "news column
    df['news'] = df['subject'] + ' ' + df['title'] + ' ' + df['text']
    df['news'] = df.apply(lambda x: x['news'].lower(), axis=1)
    df["news"] = df['news'].str.replace('[^\w\s]', '')

    return df


def naive_bayes(news_df):
    """
    Performs Multinominal Naive Bayes on the news dataset
    Export model via pickle to be used via Flask
    :param news_df:
    :return:
    """

    # Split up our data into test and train
    X_train, X_test, y_train, y_test = train_test_split(news_df['news'],
                                                        news_df['label'],
                                                        test_size=0.2)

    # Pipeline that creates a bag of words
    # Applies Multinominal Naive Bayes model
    pipeline = Pipeline([('tfidf', TfidfVectorizer(stop_words='english')),
                         ('nbmodel', MultinomialNB())])

    # Train data
    pipeline.fit(X_train, y_train)

    # Predicting the label
    prediction = pipeline.predict(X_test)
    # print(prediction)

    # Checking model performance
    print("Multinominal Naive Bayes model performance")
    print(classification_report(y_test, prediction))
    print(confusion_matrix(y_test, prediction))

    # Serialising the file
    with open('naive.pickle', 'wb') as handle:
        pickle.dump(pipeline, handle, protocol=pickle.HIGHEST_PROTOCOL)


def sgd_classifier(news_df):
    """
    Performs SGD Classification on the news dataset
    Export model via pickle to be used via Flask
    :param news_df:
    :return:
    """
    X_train, X_test, y_train, y_test = train_test_split(news_df['news'],
                                                        news_df['label'],
                                                        test_size=0.25)

    pipeline = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', SGDClassifier()),
    ])

    pipeline.fit(X_train, y_train)

    prediction = pipeline.predict(X_test)

    # Checking model performance
    print("SGD Classifier Performance")
    print(classification_report(y_test, prediction))
    print(confusion_matrix(y_test, prediction))

    with open('sgd.pickle', 'wb') as handle:
        pickle.dump(pipeline, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    file_paths = ['data/Fake.csv', 'data/True.csv']
    news_df = clean_data(file_paths)
    naive_bayes(news_df)
    sgd_classifier(news_df)
