import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn import metrics
import pickle



def clean_data(file_paths):
    fake = pd.read_csv(file_paths[0])
    real = pd.read_csv(file_paths[1])

    fake['Target'] = 1
    real['Target'] = 0

    frames = [fake, real]

    df = pd.concat(frames)

    patternDel = "http"
    filter1 = df['date'].str.contains(patternDel)

    df = df[~filter1]

    df_ = df.copy()
    df_ = df_.sort_values(by=['date'])
    df_ = df_.reset_index(drop=True)
    df_['news'] = df_['subject'] + ' ' + df_['title'] + ' ' + df_['text']
    df_['news'] = df_.apply(lambda x: x['news'].lower(), axis=1)
    df_["news"] = df_['news'].str.replace('[^\w\s]', '')

    return df, df_


def logistic_regression(df_):
    X_train, X_test, y_train, y_test = train_test_split(df_['news'],
                                                        df_['Target'],
                                                        random_state=0)

    vect = CountVectorizer().fit(X_train)
    X_train_vectorized = vect.transform(X_train)
    model = LogisticRegression()
    model.fit(X_train_vectorized, y_train)

    predictions = model.predict(vect.transform(X_test))

    print('AUC: ', roc_auc_score(y_test, predictions))

    print(metrics.confusion_matrix(y_test, predictions, labels=[0, 1]))
    # Printing the precision and recall, among other metrics
    print(metrics.classification_report(y_test, predictions, labels=[0, 1]))

if __name__ == '__main__':
    file_paths = ['data/Fake.csv', 'data/True.csv']
    df, df_ = clean_data(file_paths)
    logistic_regression(df_)

