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

    VOCAB_SIZE = 7000

    tokenized_train_texts = [tokenize(x) for x in train_texts]
    
    vocab = []

    c = Counter([])
    for txt in tokenized_train_texts:
        txt = [x for x in txt if x not in STOPWORDS]
        c += Counter(txt)
    vocab = map(lambda x: x[0], c.most_common(VOCAB_SIZE))

    vocab = set(vocab) # for faster searching
    vocab.add("UNK")

    assert "good" in vocab
    assert all(x not in vocab for x in STOPWORDS)
    assert len(vocab) == VOCAB_SIZE + 1

    def update_class_word_counter(text, class_word_counter, class_word_total):
        for token in text:
            class_word_total += 1;
            if token in vocab:
                class_word_counter[token] += 1
            else:
                class_word_counter["UNK"] += 1
        return class_word_total

    positive_class_word_counter = defaultdict(int)
    negative_class_word_counter = defaultdict(int)

    positive_word_total = 0;
    negative_word_total = 0;

    for text, label in zip(tokenized_train_texts, train_labels):
        if label == 'neg':
            negative_word_total = update_class_word_counter(text, negative_class_word_counter, negative_word_total)
        else:
            positive_word_total = update_class_word_counter(text, positive_class_word_counter, positive_word_total)

    return vocab, positive_class_word_counter, negative_class_word_counter, positive_word_total, negative_word_total, 0.5



ALPHA = 0.5

def word_probability(VOCAB_SIZE, vocab, word, class_word_counter, class_word_total):
    if word not in vocab:
        word = "UNK"
    return (ALPHA + class_word_counter[word]) / ((VOCAB_SIZE + 1) * ALPHA + class_word_total)

def classify(texts, params):
    vocab, positive_class_word_counter, negative_class_word_counter, positive_word_total, negative_word_total, prob = params
    res = []
    tokenized_texts = [tokenize(txt) for txt in texts]
    for txt in tokenized_texts:
        positive_probability = 0.0
        negative_probability = 0.0
        for word in txt:
            positive_probability += log(word_probability(len(vocab), vocab, word, positive_class_word_counter, positive_word_total))
            negative_probability += log(word_probability(len(vocab), vocab, word, negative_class_word_counter, negative_word_total))
        positive_probability += log(prob)
        negative_probability += log(1 - prob)
        if (positive_probability > negative_probability):
            res.append('pos')
        else:
            res.append('neg')
    return res