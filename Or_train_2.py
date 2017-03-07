import collections
import nltk.classify.util, nltk.metrics
from nltk.classify import NaiveBayesClassifier, MaxentClassifier, SklearnClassifier
import csv
from sklearn import cross_validation
from sklearn.svm import LinearSVC, SVC
import random
from nltk.corpus import stopwords
import itertools
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from nltk import precision
from nltk.metrics import *
from nltk.metrics.scores import (accuracy, precision, recall, f_measure, log_likelihood, approxrand)
                                          
from nltk.metrics.scores import precision
import json
import pandas as pd
import matplotlib.pyplot as plt
import re
import numpy as np

 
posdata = []
with open('positive-data.csv', 'rb') as myfile:    
    reader = csv.reader(myfile, delimiter=',')
    for val in reader:
        posdata.append(val[0])
                 
 
negdata = []
with open('negative-data.csv', 'rb') as myfile:    
    reader = csv.reader(myfile, delimiter=',')
    for val in reader:
        negdata.append(val[0])            
 
def word_split(data):    
    data_new = []
    for word in data:
        word_filter = [i.lower() for i in word.split()]
        data_new.append(word_filter)
    return data_new
 
def word_split_sentiment(data):
    data_new = []
    for (word, sentiment) in data:
        word_filter = [i.lower() for i in word.split()]
        data_new.append((word_filter, sentiment))
    return data_new
    
def word_feats(words):    
    return dict([(word, True) for word in words])
 
stopset = set(stopwords.words('english')) - set(('over', 'under', 'below', 'more', 'most', 'no', 'not', 'only', 'such', 'few', 'so', 'too', 'very', 'just', 'any', 'once'))
     
def stopword_filtered_word_feats(words):
    return dict([(word, True) for word in words if word not in stopset])
 
def bigram_word_feats(words, score_fn=BigramAssocMeasures.chi_sq, n=200):
    bigram_finder = BigramCollocationFinder.from_words(words)
    bigrams = bigram_finder.nbest(score_fn, n)
    """
    print words
    for ngram in itertools.chain(words, bigrams): 
        if ngram not in stopset: 
            print ngram
    exit()
    """    
    return dict([(ngram, True) for ngram in itertools.chain(words, bigrams)])
    
def bigram_word_feats_stopwords(words, score_fn=BigramAssocMeasures.chi_sq, n=200):
    bigram_finder = BigramCollocationFinder.from_words(words)
    bigrams = bigram_finder.nbest(score_fn, n)
    """
    print words
    for ngram in itertools.chain(words, bigrams): 
        if ngram not in stopset: 
            print ngram
    exit()
    """    
    return dict([(ngram, True) for ngram in itertools.chain(words, bigrams) if ngram not in stopset])
 
# Calculating Precision, Recall & F-measure
def evaluate_classifier(featx):
    negfeats = [(featx(f), 'neg') for f in word_split(negdata)]
    posfeats = [(featx(f), 'pos') for f in word_split(posdata)]
    negcutoff = len(negfeats)*3/4
    poscutoff = len(posfeats)*3/4
    trainfeats = negfeats[:negcutoff] + posfeats[:poscutoff]
    #testfeats = negfeats[negcutoff:] + posfeats[poscutoff:]

    
    print 'Reading Tweets\n'
    tweets_data_path = '20161019_202620.txt'
    tweets_data = []
    tweets_file = open(tweets_data_path, "r")
    for line in tweets_file:
	    try:
	        tweet = json.loads(line)
	        tweets_data.append(tweet)
	    except:
	    	continue
	     	
    tweets = pd.DataFrame()
    tweets['text'] = [tweet.get('text','') for tweet in tweets_data]
    
    tdata = tweets['text']
    negfeats = [(featx(f), 'neg') for f in word_split(tdata)]
    testfeats = negfeats

    print np.shape(testfeats)
    #testfeats = negfeats[negcutoff:] + posfeats[poscutoff:]
    #print np.shape(testfeats)
    
    
    # using 3 classifiers
    classifier_list = ['nb', 'maxent', 'svm']     
        
    for cl in classifier_list:
        if cl == 'maxent':
            classifierName = 'Maximum Entropy'
            classifier = MaxentClassifier.train(trainfeats, 'GIS', trace=0, encoding=None, labels=None,  gaussian_prior_sigma=0, max_iter = 1)
        elif cl == 'svm':
            classifierName = 'SVM'
            classifier = SklearnClassifier(LinearSVC(), sparse=False)
            classifier.train(trainfeats)
        else:
            classifierName = 'Naive Bayes'
            classifier = NaiveBayesClassifier.train(trainfeats)
            
        refsets = collections.defaultdict(set)
        testsets = collections.defaultdict(set)
 
        for i, (feats, label) in enumerate(testfeats):
                refsets[label].add(i)
                observed = classifier.classify(feats)
                testsets[observed].add(i)

        print testsets[observed]

        accuracy = nltk.classify.util.accuracy(classifier, testfeats)
        #pos_precision = nltk.metrics.precision(refsets['pos'], testsets['pos'])
        #pos_recall = nltk.metrics.recall(refsets['pos'], testsets['pos'])
        #pos_fmeasure = nltk.metrics.f_measure(refsets['pos'], testsets['pos'])
        #neg_precision = nltk.metrics.precision(refsets['neg'], testsets['neg'])
        #neg_recall = nltk.metrics.recall(refsets['neg'], testsets['neg'])
        #neg_fmeasure =  nltk.metrics.f_measure(refsets['neg'], testsets['neg'])
        
        print ''
        print '---------------------------------------'
        print 'SINGLE FOLD RESULT ' + '(' + classifierName + ')'
        print '---------------------------------------'
        print 'accuracy:', accuracy
        #print 'precision', (pos_precision + neg_precision) / 2
        #print 'recall', (pos_recall + neg_recall) / 2
        #print 'f-measure', (pos_fmeasure + neg_fmeasure) / 2    
                
        #classifier.show_most_informative_features()
    

    

        
evaluate_classifier(word_feats)
#evaluate_classifier(stopword_filtered_word_feats)
#evaluate_classifier(bigram_word_feats)    
#evaluate_classifier(bigram_word_feats_stopwords)