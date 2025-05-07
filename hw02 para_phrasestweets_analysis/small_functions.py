import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

def square_root_equi(start, end, length):
    """
    Returns a 1d numpy array of the specified length, containing the square roots of equi-distant input values
    between start and end (both included).

    >>> square_root_equi(4,9,3)
    array([2.        , 2.54950976, 3.        ])
    """
    return np.sqrt(np.linspace(start, end, num= length)) # TODO: Exercise 2.1


def odd_ones_squared(rows, columns):
    """
    Returns a 2d numpy array with shape (rows, columns). The matrix cells contain increasing numbers,
    where all odd numbers are squared.

    >>> odd_ones_squared(3,5)
    array([[  0,   1,   2,   9,   4],
           [ 25,   6,  49,   8,  81],
           [ 10, 121,  12, 169,  14]])
    """
    a = np.arange(rows * columns)
    a[1::2] = a[1::2] ** 2
    return a.reshape((rows, columns))
    # TODO: Exercise 2.2
