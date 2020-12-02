# -------------------------------------------------------------------------
'''
    Problem 2: reading data set from a file, and then split them into training, validation and test sets.

    The functions for handling data

    20/100 points
'''

import numpy as np # for linear algebra
def loadData():
    '''
        Read all labeled examples from the text files.
        Note that the data/X.txt has a row for a feature vector for intelligibility.

        n: number of features
        m: number of examples.

        :return: X: numpy.ndarray. Shape = [n, m]
                y: numpy.ndarray. Shape = [m, ]
    '''
    X = np.loadtxt("data/X.txt")
    y = np.loadtxt("data/y.txt")
    xDim = X.shape
    yDim = y.shape
    return X.T, y;


def appendConstant(X):
    '''
    Appending constant "1" to the beginning of each training feature vector.
    X: numpy.ndarray. Shape = [n, m]
    :return: return the training samples with the appended 1. Shape = [n+1, m]
    '''

    X = np.insert(arr = X, obj = 0, values = 1, axis = 0)
    return X;



def splitData(X, y, train_ratio = 0.8):
    '''
	X: numpy.ndarray. Shape = [n+1, m]
	y: numpy.ndarray. Shape = [m, ]
    split_ratio: the ratio of examples go into the Training, Validation, and Test sets.
    Split the whole dataset into Training, Validation, and Test sets.
    :return: return Training, Validation, and Test sets.
    '''
    split = int(X.shape[1]*train_ratio)
    tr_X, test_X = X[:,:split], X[:, split:]
    tr_Y, test_Y = y[:split], y[split:]
    
    return (tr_X, tr_Y), (test_X, test_Y)

