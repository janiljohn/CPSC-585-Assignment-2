import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords 
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import matplotlib.pyplot as plt


def bag_of_wordsify(dataset, feature_functions=[], max_token_features=1000):
    nltk.download('punkt')
    nltk.download('stopwords')

    cleaned_texts = []

    custom_features = [[] for _ in feature_functions]

    # Preprocess the text data
    stop_words = set(stopwords.words('english'))
    for text in dataset['text']:
        text = re.sub('[^A-Za-z]', ' ', text).lower()
        words = word_tokenize(text)
        words = [word for word in words if word not in stop_words]

        cleaned_text = ' '.join(words)
        cleaned_texts.append(cleaned_text)
        # print(cleaned_text)

        for i, func in enumerate(feature_functions):
            custom_features[i].append(func(text))

    # Vectorize the cleaned text data
    vectorizer = CountVectorizer(max_features=max_token_features)
    _thing = vectorizer.fit_transform(cleaned_texts)
    X = _thing.toarray()

    for feature in custom_features:
        feature_array = np.array(feature).reshape(-1, 1)
        X = np.hstack((X, feature_array))

    feature_names = np.append(vectorizer.get_feature_names_out(), 
                              ['custom_feature_' + str(i) for i in range(len(feature_functions))])

    return X, feature_names


# MAIN FILE
def contains_not(text):
    return 1 if 'not' in text.split() else 0

def contains_security(text):
    return 1 if 'security' in text.split() else 0
