import random
import os
import re
from string import punctuation
from collections import defaultdict
from copy import deepcopy
from collections import Counter
from math import log

def tokenize(text):
    """
    Preprocesses text and split it into the list of words
    :param: text(str): movie review
    """
    text = text.lower()
    #p = re.compile(r'<.*?>')
    #tokens = re.findall('[\'\w]+', p.sub('',text))
    tokens = re.findall('\w+',text)
    return tokens

STOPWORDS = ['i','me','my','myself','we','our','ours','ourselves','you','your','yours','yourself','yourselves','he','him','his','himself','she','her','hers','herself','it','its','itself','they','them','their','theirs','themselves','what','which','who','whom','this','that','these','those','am','is','are','was','were','be','been','being','have','has','had','having','do','does','did','doing','a','an','the','and','but','if','or','because','as','until','while','of','at','by','for','with','about','against','between','into','through','during','before','after','above','below','to','from','up','down','in','out','on','off','over','under','again','further','then','once','here','there','when','where','why','how','all','any','both','each','few','more','most','other','some','such','no','nor','not','only','own','same','so','than','too','very','s','t','can','will','just','I','should','now','d','ll','m','o','re','ve','y','ain','aren','couldn','didn','doesn','hadn','hasn','haven','isn','ma','mightn','mustn','needn','shan','shouldn','wasn','weren','won','wouldn']
STOPWORDS = set(STOPWORDS)

def train(train_texts, train_labels):
    """
    Trains classifier on the given train set represented as parallel lists of texts and corresponding labels.
    :param train_texts: a list of texts (str objects), one str per example
    :param train_labels: a list of labels, one label per example
    :return: learnt parameters, or any object you like (it will be passed to the classify function)
    """

    VOCAB_SIZE = 3000

    tokenized_train_texts = [tokenize(x) for x in train_texts]
    
    vocab = []

    c = Counter([])
    for txt in tokenized_train_texts:
        txt = [x for x in txt if x not in STOPWORDS]
        c += Counter(txt)
    vocab = map(lambda x: x[0], c.most_common(VOCAB_SIZE))

    vocab = set(vocab) # for faster searching
    vocab.add("UNK")


    positive_train_vectors = []
    negative_train_vectors = []
    for text, label in zip(tokenized_train_texts, train_labels):
        t_dict = defaultdict(int)
        for token in text:
            if token in vocab:
                t_dict[token] = 1
            else:
                t_dict['UNK'] = 1
        if label == 'pos':
            positive_train_vectors.append(t_dict)
        else:
            negative_train_vectors.append(t_dict)


    def word_probability(word, vectors):
        class_total = 0
        for text in vectors:
            class_total += text[word]
        
        return  (class_total + 1) / (len(vectors) + 2) 


    pos_words_probs = defaultdict(float)
    neg_words_probs = defaultdict(float)
    for word in vocab:
        pos_words_probs[word] = word_probability(word, positive_train_vectors)
        neg_words_probs[word] = word_probability(word, negative_train_vectors)

    return vocab, pos_words_probs, neg_words_probs

def classify(texts, params):
    """
    Classify texts given previously learnt parameters.
    :param texts: texts to classify
    :param params: parameters received from train function
    :return: list of labels corresponding to the given list of texts
    """
    vocab, pos_words_probs, neg_words_probs = params
    vocab = params[0]
    pos_words_probs = params[1]
    neg_words_probs = params[2]

    res = []
    tokenized_texts = [tokenize(txt) for txt in texts]
    for txt in tokenized_texts:
        positive_probability = 0.0
        negative_probability = 0.0
        for word in vocab:
            contains = 0
            if word in txt:
                contains = 1
            positive_probability += log(pos_words_probs[word] * contains + (1 - pos_words_probs[word]) * (1 - contains))
            negative_probability += log(neg_words_probs[word] * contains + (1 - neg_words_probs[word]) * (1 - contains))
        if (positive_probability > negative_probability):
            res.append('pos')
        else:
            res.append('neg')
    return res