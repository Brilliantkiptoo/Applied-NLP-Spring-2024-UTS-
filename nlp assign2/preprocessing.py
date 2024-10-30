import pandas as pd
import nltk
from gensim.models.doc2vec import TaggedDocument
from sklearn.model_selection import train_test_split

nltk.download('punkt')

def preprocess_text(text):
    # Example of text preprocessing steps for tokenizing
    tokens = nltk.word_tokenize(text.lower())
    return tokens

def preprocess_data(df, text_column, target_column):
    # Process the text data and split into train-test sets
    df[text_column] = df[text_column].apply(preprocess_text)
    X_train, X_test, y_train, y_test = train_test_split(df[text_column], df[target_column], test_size=0.2)
    return X_train, X_test, y_train, y_test
