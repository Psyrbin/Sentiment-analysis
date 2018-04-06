import random
import os
import re
import numpy as np
from scipy.sparse import csr_matrix
from collections import defaultdict
from math import log
from math import exp

def tokenize(text):
    text = text.lower()
    #p = re.compile(r'<.*?>')
    #tokens = re.findall('[\'\w]+', p.sub('',text))
    tokens = re.findall('\w+',text)
    return tokens

def get_matrix(dictionary, texts):
    indptr = [0]
    indices = []
    data = []
    for text in texts:
        indices.append(0)
        data.append(1)
        bigrams = [x for x in zip(text[:-1], text[1:])]
        for token in text:
            if token in dictionary:
                indices.append(dictionary[token])
                data.append(1)
        for bigram in bigrams:
            if bigram in dictionary:
                indices.append(dictionary[bigram])
                data.append(1)
        indptr.append(len(indices))
    return csr_matrix((data, indices, indptr), shape = (len(texts), len(dictionary) + 1))

def accuracy(w, X, y):
    predictions = sigmoid(X.dot(w)) >= 0.5
    return np.sum(predictions == y) / y.shape[0]




def sigmoid(x):
    return 1 / (1 + np.exp(np.negative(x)))

def initialize_weights(size):
    return np.zeros(size)

def loss(w, X, y, alpha):
    N, _ = X.shape
    h = sigmoid(X.dot(w))
    for idx, elem in enumerate(h):
        if elem == 0:
            h[idx] = 0.0000000001
        elif elem == 1:
            h[idx] = 0.9999999999
    L = np.multiply(y, np.log(h)) + np.multiply((1 - y), np.log(1 - h))
    return -np.sum(L) / N + alpha * np.sum(np.power(w[1:], 2))


def gradient(w, X, y, alpha):
    N, M = X.shape
    grad = np.zeros(M)
    tmp = sigmoid(X.dot(w)) - y
    
    reg = 2 * w * alpha
    reg[0] = 0

    grad = X.transpose().dot(tmp) + reg
    grad = np.divide(grad, N)
    return grad

def update_weights(w, grad, lr):
    return w - grad * lr



LEARNING_RATE = 0.04
REG_PARAM = 0.00005
def train(train_texts, train_labels):
    vocab = []
    tokenized_train_texts = [tokenize(text) for text in train_texts]
    for text in tokenized_train_texts:
        bigrams = [x for x in zip(text[:-1], text[1:])]
        for token in text:
            vocab.append(token)
        for bigram in bigrams:
            vocab.append(bigram)
    vocab = list(set(vocab))

    dictionary = defaultdict(int)
    word_num = 1
    for word in vocab:
        dictionary[word] = word_num
        word_num += 1

    training_set = get_matrix(dictionary, tokenized_train_texts)
    weights = initialize_weights(len(dictionary) + 1)

    labels = []
    for label in train_labels:
        if label == 'pos':
            labels.append(1)
        else:
            labels.append(0)
    labels = np.array(labels)


    loss_cur = loss(weights, training_set, labels, REG_PARAM)
    grad = gradient(weights, training_set, labels, REG_PARAM)
    loss_prev = loss_cur + 1
    iter_num = -1

    while loss_prev - loss_cur > 0.05 or iter_num < 5000:
        iter_num += 1
        if iter_num % 1000 == 0:
            loss_prev = loss_cur
            loss_cur = loss(weights, training_set, labels, REG_PARAM)
        weights = update_weights(weights, grad, LEARNING_RATE)
        grad = gradient(weights, training_set, labels, REG_PARAM)
    return dictionary, weights

def classify(texts, params):
    dictionary, weights = params
    tokenized_texts = [tokenize(text) for text in texts]
    test_set = get_matrix(dictionary, tokenized_texts)
    predictions = sigmoid(test_set.dot(weights)) >= 0.5
    res = []
    for pred in predictions:
        if pred:
            res.append('pos')
        else:
            res.append('neg')
    return res
