#!/usr/bin/env python3
"""
Creates a bag of words embedding matrix
"""
from sklearn.feature_extraction.text import CountVectorizer


def bag_of_words(sentences, vocab=None):
    """
    Creates a bag of words embedding matrix
    sentences: a list of sentences to analyze
    vocab: a list of the vocabulary words to use for the analysis
    """
    vector = CountVectorizer(vocabulary=vocab)
    X = vector.fit_transform(sentences)
    features = vector.get_feature_names()
    embeddings = X.toarray()

    return embeddings, features
