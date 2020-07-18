import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import MultinomialNB


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





if __name__ == '__main__':
    file_paths = ['data/Fake.csv', 'data/True.csv']
    df, df_ = clean_data(file_paths)
    # # logistic_regression(df_)
    # naive_bayes(df_)
