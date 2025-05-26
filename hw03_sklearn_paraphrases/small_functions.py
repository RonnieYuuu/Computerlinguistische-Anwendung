import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

def trigram_quadragram_vectorizer(texts):
    """
    Takes a list of text strings and returns a CountVectorizer that considers all trigrams and quadragrams that occur
    in at least 3 texts.

    >>> l = ["My name is Bond", "His name is not Bond", "I think your name is Bond", "I think my name is Bond", "You think my name is Bond"]
    >>> v = trigram_quadragram_vectorizer(l)
    >>> v.get_feature_names_out()
    ['my name is', 'my name is bond', 'name is bond']
    """
    vectorizer = CountVectorizer(ngram_range=(3, 4), lowercase=True)

    X = vectorizer.fit_transform(texts)

    binary_matrix = (X > 0).astype(int)

    ngram_counts = np.array(binary_matrix.sum(axis=0)).flatten()

    mask = ngram_counts >= 3
    ngram_features = np.array(vectorizer.get_feature_names_out())
    selected_ngrams = ngram_features[mask]

    final_vectorizer = CountVectorizer(vocabulary=selected_ngrams)
    final_vectorizer.fit(texts)

    return final_vectorizer  # TODO: Exercise 1

