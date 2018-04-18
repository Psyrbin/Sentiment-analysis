import random
import os
import re
import numpy as np
from scipy.sparse import csr_matrix
from collections import defaultdict
from math import log
from math import exp


def loadGlove(file):
    f = open(file,'r')
    glove = {}
    for entry in f:
        entry = entry.split()
        word = entry[0]
        vector = np.array([float(val) for val in entry[1:]])
        glove[word] = vector
    return glove


def tokenize(text):
    text = text.lower()
    tokens = re.findall('[a-z]+',text)
    return tokens

def vectorize(text, glove):
    res = np.zeros(glove['the'].shape)
    count = 0
    for token in text:
        if token in glove:
            res += glove[token]
            count += 1
    return np.concatenate((np.ones(1), res / count))

def sigmoid(x):
    return 1 / (1 + np.exp(np.negative(x)))

def tanh_der(x):
    return 1 - np.power(np.tanh(x), 2)

def initialize_weights(dim1, dim2):
    return np.random.randn(dim1, dim2) * 0.001

def loss(X, y, W1, W2, alpha):
    N, M = X.shape
    a1 = np.tanh(np.dot(W1, X.T))
    a1 = np.concatenate((np.ones((1, a1.shape[1])), a1))
    h = sigmoid(np.dot(W2, a1)) #[h(x1) h(x2) ... h(xn)]

    for elem in np.nditer(h, op_flags=['readwrite']):
        if elem == 0:
            elem[...] = 0.0000001
        elif elem == 1:
            elem[...] = 0.9999999

    tmp = np.multiply(y, np.log(h))
    tmp += np.multiply((1 - y), np.log(1 - h))
    L = -np.sum(tmp) / N

    reg = np.sum(np.power(W1[:,1:],2)) + np.sum(np.power(W2[:,1:],2)) 
    L += alpha * reg
    return L


def feedforward(x, W1, W2):
    z1 = np.dot(W1, x.T)
    a1 = np.tanh(z1)
    a1 = np.concatenate((np.ones((1, 1)), a1))

    return z1, a1, sigmoid(np.dot(W2, a1))

def backpropagation(X, y,  W1, W2, alpha):
    N, M = X.shape
    W1_grad = np.zeros(W1.shape)
    W2_grad = np.zeros(W2.shape)
    for k in range(N):
        z1, a1, a2 = feedforward(X[k,:], W1, W2)
        delta_2 = a2 - y[k]
        delta_1 = np.dot(W2.T, delta_2)[1:]
        delta_1 = np.multiply(delta_1, tanh_der(z1))

        W1_grad +=  np.dot(delta_1, X[k,:]) ## transpose something
        W2_grad += np.dot(delta_2, a1.T)
    W1_grad /= N
    W2_grad /= N

    W1_reg = W1 * alpha * 2
    W1_reg[:, 0] = 0
    W2_reg = W2 * alpha * 2
    W2_reg[:, 0] = 0    

    W1_grad += W1_reg
    W2_grad += W2_reg
    return W1_grad, W2_grad

def update_weights(W1, W2, W1_grad, W2_grad, lr):
    return W1 - W1_grad * lr, W2 - W2_grad * lr

def get_batch(num, X, y, size):
    N,_ = X.shape
    start = (num-1) * size
    if size * num > N:
        return X[start :, :], y[start :]
    else:
        end = num * size
        return X[start : end, :], y[start : end]



def train(train_texts, train_labels):
    LAYER_SIZE = 200
    EMBEDDING_SIZE = 300
    REG_PARAM = 0.0004
    LEARNING_RATE = 0.4004
    BATCH_SIZE = 25000

    labels = []
    for label in train_labels:
        if label == 'pos':
            labels.append(1)
        else:
            labels.append(0)
    labels = np.array(labels)
    
    glove = loadGlove('glove.6B.300d.txt')
    tokenized_train_texts = [tokenize(x) for x in train_texts]
    vectorized_train_texts = [vectorize(x, glove) for x in tokenized_train_texts]
    train_set = np.matrix(vectorized_train_texts)
    

    W1 = initialize_weights(LAYER_SIZE, EMBEDDING_SIZE + 1)
    W2 = initialize_weights(1, LAYER_SIZE + 1)
    W1_p = W1
    W2_p = W2
    i = 0
    loss_cur = 100
    flag = False
    N,_ = train_set.shape
    batch_count = N // BATCH_SIZE

    while loss_cur >= 0.507:
        for batch_n in range(batch_count):

            X_cur, y_cur = get_batch(batch_n + 1, train_set, labels, BATCH_SIZE)
            if i % (batch_count * 10) == 0:
                if i % (batch_count * 90) == 0:
                    LEARNING_RATE *= 1.07
                
                if not flag:
                    loss_prev = loss_cur
                loss_cur = loss(train_set, labels, W1, W2, REG_PARAM)
                #print(loss_cur)
                if(loss_cur > loss_prev):
                    W1 = W1_p
                    W2 = W2_p
                    LEARNING_RATE /= 1.2
                    #print("Went back")
                    flag = True
                else:
                    W1_p = W1
                    W2_p = W2
                    flag = False
            g1, g2 = backpropagation(X_cur, y_cur, W1, W2, REG_PARAM)
            W1, W2 = update_weights(W1, W2, g1, g2, LEARNING_RATE)
            i += 1
    return glove, W1, W2



def classify(texts, params):
    glove, W1, W2 = params
    tokenized_texts = [tokenize(x) for x in texts]
    vectorized_texts = [vectorize(x, glove) for x in tokenized_texts]
    test_set = np.matrix(vectorized_texts)

    res = []
    for i, text in enumerate(vectorized_texts):
        _, _, pred = feedforward(test_set[i,:], W1, W2)
        pred = np.sum(pred)
        if pred >= 0.5:
            res.append('pos')
        else:
            res.append('neg')
    return res
