import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  num_class = W.shape[1]
  scores = X.dot(W)   # (N, C)

  # for numerical stability
  scores -= np.max(scores)
  for i in range(num_train):
    numer = np.exp(scores[i, :])  # (1, C)
    denum = np.sum(numer)   # scaler
    loss += -1 * scores[i, y[i]] + np.log(denum)

    dW += X[i, :][:, np.newaxis].dot(numer[np.newaxis, :]) / denum  # (D, C)
    dW[:, y[i]] -= X[i, :].T    # (D, 1)
    
  loss /= num_train
  loss += reg * np.sum(W * W)

  dW /= num_train
  dW += 2 * reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  scores = X.dot(W)   # (N, C)

  # for numerical stability
  scores -= np.max(scores)
 
  correct_score_sum = scores[range(len(y)), y].sum()
  numer = np.exp(scores)   # (N, C)
  denum = np.sum(numer, axis=1)    # (N,)
  loss = -1 * correct_score_sum + np.sum(np.log(denum))
  
  multiplier = numer / denum[:, np.newaxis]   # (N, C)/(N, 1) (broadcasting)
  multiplier[range(num_train), y] -= 1 ###### NOTE I refer to yunjey@github's code for this line
  dW = X.T.dot(multiplier)    # (D, N)x(N, C) = (D, C)

  loss /= num_train
  loss += reg * np.sum(W * W)

  dW /= num_train
  dW += 2 * reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

