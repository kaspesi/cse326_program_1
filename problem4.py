# -------------------------------------------------------------------------
'''
    Problem 4: compute sigmoid(Z), the loss function, and the gradient.
    This is the vectorized version that handle multiple training examples X.

    20/100 points
'''

import numpy as np # linear algebra
from scipy.sparse import diags
from scipy.sparse import csr_matrix

def linear(theta, X):
    '''
    theta: (n+1) x 1 column vector of model parameters
    x: (n+1) x m matrix of m training examples, each with (n+1) features.
    :return: inner product between theta and x
    '''
    return np.dot(theta.T, X);

def sigmoid(Z):
    '''
    Z: 1 x m vector. <theta, X>
    :return: A = sigmoid(Z)
    '''
    return 1 / (1 + np.exp(-Z));

def loss(A, Y):
    '''
    A: 1 x m, sigmoid output on m training examples
    Y: 1 x m, labels of the m training examples

    You must use the sigmoid function you defined in *this* file.

    :return: mean negative log-likelihood loss on m training examples.
    '''
    m = A.shape[0]
    nll = -1*(Y*(np.log(A)) + (1-Y)*(np.log(1-A)))
    return np.mean(nll)
   
def dZ(Z, Y):
    '''
    Z: 1 x m vector. <theta, X>
    Y: 1 x m, label of X

    You must use the sigmoid function you defined in *this* file.

    :return: 1 x m, the gradient of the negative log-likelihood loss on all samples wrt z.
    '''

    #G = np.zeros(Z.shape)
    gll = -1*(Y*(1/(np.exp(Z)+1)) - (1-Y)*(np.exp(Z)/(np.exp(Z)+1)) )
    return gll;

def dtheta(Z, X, Y):
    '''
    Z: 1 x m vector. <theta, X>
    X: (n+1) x m, m example feature vectors.
    Y: 1 x m, label of X
    :return: (n+1) x 1, mean of the gradient of the negative log-likelihood loss on all samples wrt theta.
    '''
    (n,m) = Z.shape
    sig = sigmoid(Z)
    ans = -(Y-sig)*X
    temp = np.mean(ans, axis=1).T
    return np.reshape(temp, (temp.shape[0],1));

def Hessian(Z, X):
    '''
    Compute the Hessian matrix on m training examples.
    Z: 1 x m vector. <theta, X>
    X: (n+1) x m, m example feature vectors.
    :return: the Hessian matrix of the negative log-likelihood loss wrt theta
    '''
    m = Z.shape[1]
    S = np.zeros(shape=(m,m))
    row,col = np.diag_indices_from(S)
    z = np.multiply(sigmoid(Z),sigmoid(-Z))
    print(z)
    print(sigmoid(Z)*sigmoid(-Z))
    print(Z)
    S[row,col] = z
    return (np.dot(np.dot(X, S), X.T))

