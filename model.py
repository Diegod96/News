import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle


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
    fake['label'] = 1
    real['label'] = 0

    # Merge both datasets
    frames = [fake, real]
    df = pd.concat(frames)

    # Delete any "http"
    patternDel = "http"
    filter1 = df['date'].str.contains(patternDel)

    # Apply filter
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
    Performs Multinominal Naive Bayes on the news dataset that is passed to it
    Make a pickle file of the model
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

    # Checking model performance
    print(classification_report(y_test, prediction))
    print(confusion_matrix(y_test, prediction))

    # Serialising the file
    with open('model.pickle', 'wb') as handle:
        pickle.dump(pipeline, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    file_paths = ['data/Fake.csv', 'data/True.csv']
    news_df = clean_data(file_paths)
    naive_bayes(news_df)
