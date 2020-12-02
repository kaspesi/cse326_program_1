# -------------------------------------------------------------------------
'''
    Problem 5: Gradient Descent and Newton method Training of Logistic Regression
    20/100 points
'''

import problem3
import problem4
from problem2 import *
import numpy as np # linear algebra
import pickle

def batch_gradient_descent(X, Y, X_test, Y_test, num_iters = 1000, lr = 0.01, log=True):
    '''
    Train Logistic Regression using Gradient Descent
    X: d x m training sample vectors
    Y: 1 x m labels
    X_test: test sample vectors
    Y_test: test labels
    num_iters: number of gradient descent iterations
    lr: learning rate
    log: True if you want to track the training process, by default True
    :return: (theta, training_log)
    training_log: contains training_loss, test_loss, and norm of theta
    '''
    
    (d,m) = X.shape
    theta = np.zeros(shape=(d, 1))
    training_log = []
    for i in range(1,num_iters):
        Z = problem4.linear(theta, X)
        te_Z = problem4.linear(theta, X_test)
        dtheta = (problem4.dtheta(Z, X, Y))
        theta = theta - (lr*dtheta)
        training_log.append((problem4.loss(problem4.sigmoid(Z), Y),
                            problem4.loss(problem4.sigmoid(te_Z), Y_test),
                            np.linalg.norm(theta)**2))
    return (theta, training_log)




def stochastic_gradient_descent(X, Y, X_test, Y_test, num_iters = 50, lr = 0.01, log=True):
    '''
    Train Logistic Regression using Gradient Descent
    X: d x m training sample vectors
    Y: 1 x m labels
    X_test: test sample vectors
    Y_test: test labels
    num_iters: number of gradient descent iterations
    lr: learning rate
    log: True if you want to track the training process, by default True
    :return: (theta, training_log)
    training_log: contains training_loss, test_loss, and norm of theta
    '''
   
    (d, m) = X.shape
    theta = np.zeros(shape=(d))
    training_log = []
    for i in range(1,num_iters):
        trainingLoss = 0
        testLoss = 0
        for j in range(1, X.shape[1]):
            Z = problem3.linear(theta, X[:,j])
            dtheta = problem3.dtheta(Z, X[:,j], Y[j])
            theta = theta - (lr*dtheta)
            trainingLoss = trainingLoss + problem3.loss(problem3.sigmoid(Z), Y[j])

        te_Z = problem4.linear(theta, X_test)
        testLoss += problem4.loss(problem3.sigmoid(te_Z), Y_test)
        trainingLoss = trainingLoss/X.shape[1]
        norm = np.linalg.norm(theta)**2
        training_log.append((trainingLoss, testLoss, norm))
    return (theta, training_log)



def Newton_method(X, Y, X_test, Y_test, num_iters = 50, log=True):
    '''
    Train Logistic Regression using Gradient Descent
    X: d x m training sample vectors
    Y: 1 x m labels
    X_test: test sample vectors
    Y_test: test labels
    num_iters: number of gradient descent iterations
    log: True if you want to track the training process, by default True
    :return: (theta, training_log)
    training_log: contains training_loss, test_loss, and norm of theta
    '''
    #########################################
    ## INSERT YOUR CODE HERE
    #########################################


# --------------------------
def train_SGD(**kwargs):
    #########################################
    ## INSERT YOUR CODE HERE
    #########################################
    # use functions defined in problem3.py to perform stochastic gradient descent

    tr_X = kwargs['Training X']
    tr_y = kwargs['Training y']
    te_X = kwargs['Test X']
    te_y = kwargs['Test y']
    num_iters = kwargs['num_iters']
    lr = kwargs['lr']
    log = kwargs['log']
    return stochastic_gradient_descent(tr_X, tr_y, te_X, te_y, num_iters, lr, log)


# --------------------------
def train_GD(**kwargs):

    # use functions defined in problem4.py to perform batch gradient descent

    tr_X = kwargs['Training X']
    tr_y = kwargs['Training y']
    te_X = kwargs['Test X']
    te_y = kwargs['Test y']
    num_iters = kwargs['num_iters']
    lr = kwargs['lr']
    log = kwargs['log']
    return batch_gradient_descent(tr_X, tr_y, te_X, te_y, num_iters, lr, log)

# --------------------------
def train_Newton(**kwargs):
    #########################################
    ## INSERT YOUR CODE HERE
    #########################################
    tr_X = kwargs['Training X']
    tr_y = kwargs['Training y']
    te_X = kwargs['Test X']
    te_y = kwargs['Test y']
    num_iters = kwargs['num_iters']
    log = kwargs['log']
    return Newton_method(tr_X, tr_y, te_X, te_y, num_iters, log)


if __name__ == "__main__":
    '''
    Load and split data, and use the three training methods to train the logistic regression model.
    The training log will be recorded in three files.
    The problem5.py will be graded based on the plots in plot_training_log.ipynb (a jupyter notebook).
    You can plot the logs using the "jupyter notebook plot_training_log.ipynb" on commandline on MacOS/Linux.
    Windows should have similar functionality if you use Anaconda to manage python environments.
    '''
    X, y = loadData()
    X = appendConstant(X)
    (tr_X, tr_y), (te_X, te_y) = splitData(X, y)

    kwargs = {'Training X': tr_X,
              'Training y': tr_y,
              'Test X': te_X,
              'Test y': te_y,
              'num_iters': 1000,
              'lr': 0.01,
              'log': True}

    theta, training_log = train_SGD(**kwargs)
    with open('./data/SGD_outcome.pkl', 'wb') as f:
        pickle.dump((theta, training_log), f)

    theta, training_log = train_GD(**kwargs)
    with open('./data/batch_outcome.pkl', 'wb') as f:
        pickle.dump((theta, training_log), f)

    # Im not a graduate student    
    # theta, training_log = train_Newton(**kwargs)
    # with open('./data/newton_outcome.pkl', 'wb') as f:
    #     pickle.dump((theta, training_log), f)


