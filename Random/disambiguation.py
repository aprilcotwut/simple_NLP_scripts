#!/usr/bin/env python
# encoding: utf-8

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# disambiguation.py: This script disambiguates a few word pairs using
#   a Naive Bayes classifier. This script is acompanied by a similarly
#   named R script which classifies wordpairs using wildcard ML
#   classifiers, including some not discussed in class "for the lols"
#
# Author: April Walker
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
import numpy as np
import pandas as pd
import ipdb
import math
import time
import nltk
from nltk.corpus import stopwords as stopwords
import sklearn
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

stop_words = set(stopwords.words('english'))

file = open('amazon_reviews.txt', 'r', encoding='utf-8')
raw = file.read()

#
# save_to_file: simple function saves array of text to file with new line
#   between elements, snagged from interwebs and changed a tad

#
#   input--
#   filename: contains output filename/path (str)
#   text: array of text to write to file (array)
def save_to_file(filename, text):
    with open(filename, mode='wt', encoding='utf-8') as myfile:
        for line in text:
            myfile.write(line)
            myfile.write('\n')

#
# clean_corpus: minimally preprocesses corpus per nltk parser given by S. Gauch
#   input--
#   corpus: raw text to parse (str)
#
#   output--
#   words: an array of strings conatining the words in the corpus (list)
def clean_corpus(corpus):
    # just in case, this checks all chars are ascii
    corpus = ''.join(char for char in corpus if ord(char)<128)
    # tokenize into words
    tokens = nltk.word_tokenize(corpus)
    # make all chars lowercase, remove non-alpha chars
    words = [word.lower() for word in tokens if word.isalpha()]
    # get frequency of words for below operation
    freq = nltk.FreqDist(words)
    # remove words only occuring once
    words = [word for word in words if freq[word] > 1]

    filtered_words = []
    for word in words:
        if word not in stop_words:
            filtered_words.append(word)

    return filtered_words

#
# get_ngrams: creates hashtable of ngrams (frequency table for n = 1)
#   input--
#   words: list containing words to consider (list)
#   n: n value for n-gram (int)
#   dict: optional existing hashtable to append to (dict)
#
#   output--
#   ngram_dict: the sorted dict of ngrams as a list (list)
def get_ngrams(words, n, ngram_dict = {}):
    for i in range(0, (len(words)-(n-1))):
        # get new ngram
        new_ngram = ' '.join(words[i:i+n])
        # if ngram exists in dictionary add occurance
        if new_ngram in ngram_dict.keys():
            ngram_dict[new_ngram] += 1
        # else add to dictionary
        else:
            ngram_dict[new_ngram] = 1
    # sort decending by occurance and return list
    ngram_dict = sorted(ngram_dict.items(), key=lambda x:-x[1])
    return ngram_dict

#
# wordpair_context: grabs conext and saves train/test from corpus for wordpairs
#   input--
#   wordpairs: contains all words for which context is wanted (array of tuples)
#   words: an array of strings conatining the words in the corpus (list)
#   n: gives the width for which context is wanted (int)
def wordpair_context(wordpairs, words, n):
    # get context: internal function to get context for specific word
    # inputs: my_word (str), words (array), n (int)
    # outputs: context (array)
    def get_context(my_word, words, n):
        # note word is skipped if appropropriate context not avaliable
        context = []
        for i in range(n, (len(words)-(n-1))):
            if words[i] == my_word:
                # get context for instance of word
                tmp = words[i-n:i+(n+1)]
                tmp.pop(n)
                # append context for this instance to array
                context.append(' '.join(tmp))
        return context

    for pair in wordpairs:
        for my_word in pair:
            # declare filenames
            train_file = '_'.join([my_word, 'training.txt'])
            test_file = '_'.join([my_word, 'testing.txt'])
            # get context
            context = get_context(my_word, words, n)
            # train/test split
            msk = np.random.rand(len(context)) < 0.8
            train = np.array(context)[msk]
            test = np.array(context)[~msk]
            # save train/test sets
            save_to_file(train_file, train)
            save_to_file(test_file, test)

#
# calc_cond_prob: calculates the conditional probabilites for all words in dict
#   input--
#   counts_dict: dictionary of word counts(dict)
#   c: total number words in corpus (int)
#   V: total size of volcabulary
#
#   output--
#   bigram_dict: dictionary of bigrams with pmi (dict)
def calc_cond_prob(counts_dict, cond_counts_dict, c, V):
    keys = [*counts_dict]
    for key in keys:
        if key in cond_counts_dict.keys():
            counts_dict[key] = ((cond_counts_dict[key] + 1)/(c + V))
        else:
            counts_dict[key] = (1/(c + V))
    # add a default value
    counts_dict['-1'] = (1/(c + V))
    return counts_dict

#
# get_classification: calculates the predicted probabilites and final decided class
#   of our testing dataset using training probabilites and Naive Bayes
#   input--
#   data: likely, a testing dataset to be classified (pd.DataFrame)
#   cond_prob_a/b: contains the conditional probailites of the training set
#   prior_a/b: contains the probaility of class a and b respectively (double)
#   pair: contains the two words being disambiguated (tuple of strs)
#
#   output--
#   classification: dataframe with class and specicifc probabilites (pd.DataFrame)
def get_classification(data, cond_prob_a, cond_prob_b, prior_a, prior_b, pair):
    classification = pd.DataFrame(columns = ["pred", pair[0], pair[1]])
    n = len(data)

    for i in range(0, n):
        keys = data.iloc[i]
        # reset probability to prior before each new key
        prob_a = prior_a
        prob_b = prior_b
        for key in keys:
            # if key is in our previous dataset
            if key in cond_prob_a:
                # given a prior, sum the conditional probabilities of each word
                prob_a = prob_a*cond_prob_a[key]
                prob_b = prob_b*cond_prob_b[key]

            # else use default
            else:
                prob_a = prob_a*cond_prob_a['-1']
                prob_b = prob_b*cond_prob_b['-1']

        if (prob_a > prob_b):
            classification.loc[i] = (pair[0], prob_a, prob_b)
        else:
            classification.loc[i] = (pair[1], prob_a, prob_b)
    return(classification)


def run_test():
    file = open('test.log', 'w')
    # test naive bayes using example from class
    context_0 = nltk.word_tokenize("fish smoked fish fish line fish haul smoked")
    context_1 = nltk.word_tokenize("guitar jazz line")

    pair = ("bassfish", "bassmusic")

    # get counts for each word present in training sets
    counts_0 = get_ngrams(context_0, 1, ngram_dict = {})
    counts_1 = get_ngrams(context_1, 1, ngram_dict = {})

    c_0 = len(context_0)
    c_1 = len(context_1)

    tokens = context_0 + context_1
    counts = get_ngrams(tokens, 1)
    # gets the length of the volcabulary of the training set
    V = len(counts)

    # a dictionary with the conditional probabilities for each word
    cond_prob_0 = calc_cond_prob(dict(counts), dict(counts_0), c_0, V)
    cond_prob_1 = calc_cond_prob(dict(counts), dict(counts_1), c_1, V)

    # prior class probabilites
    prob_0 = .75
    prob_1 = .25

    # # # Step 4: Testing our classifier
    new = nltk.word_tokenize("line guitar jazz jazz")
    test = pd.DataFrame([new])
    pred = get_classification(test, cond_prob_0, cond_prob_1, prob_0, prob_1, pair)

    file.write(str(pred))
    file.close()




# # # # # # # # # # # # # # # #
### Main Script Starts Here ###
# # # # # # # # # # # # # # # #

run_test()

# # # Clean the corpus
print('Beginning preprocessing...')
start = time.time()
# words = clean_corpus(raw)
end = time.time()
print((end - start), 'seconds elapsed')

# # # Step 1: Extract and save wordpairs

# indicate wordpairs of interest here
wordpairs = [('night', 'seat'),
             ('kitchen', 'cough'),
             ('car', 'bike'),
             ('manufacturer', 'bike'),
             ('big', 'small'),
             ('huge', 'heavy')]

# extract and save (only call this once...)
# wordpair_context(wordpairs, words, 5)

# # #
# This loop contains the dataset prep AND classifer training/testing
# # #
for pair in wordpairs:
    # # #
    # Step 2: Prep out train/test datsets - NOTE: I know the code as written
    #  is weird for a Naive Bayes classifer, originally I was going to experiment
    #  with classifiers in python then last minute switched to R, thus I formatted
    # the train/test set more "traditionally" with the target.
    # # #
    train = pd.DataFrame()
    test = pd.DataFrame()
    truth = pd.DataFrame() # a copy of the test dataset with the true target
    for word in pair:
        # declare filenames to grab context
        train_file = '_'.join([word, 'training.txt'])
        test_file = '_'.join([word, 'testing.txt'])
        # get context
        new_train = pd.read_csv(train_file, delimiter=' ', header=None)
        new_test = pd.read_csv(test_file, delimiter=' ', header=None)
        # append true target value
        new_train['target'] = word
        new_test['target'] = word
        # append dataframes to train/truth sets
        train = train.append(new_train)
        truth = truth.append(new_test)

    # shuffle our dataframes once both wordpairs are present
    train = train.sample(frac=1)
    truth = truth.sample(frac=1)

    # drop our known target for our test set
    test = truth.drop(['target'], axis=1)

    # # #
    # Step 3: Training our classifiers. On top our standard naive bayes I'm
    #   going to try out some wild card ML classifiers in R.
    # # #

    # re-split your train dataset and drop target since it's implied
    train_0 = train.loc[train['target'] == pair[0]].iloc[:,:-1]
    train_1 = train.loc[train['target'] == pair[1]].iloc[:,:-1]

    # get total number of tokens in each training sets
    c_0 = train_0.shape[0]*train_0.shape[1]
    c_1 = train_1.shape[0]*train_1.shape[1]
    # get counts for each word present in training sets
    counts_0 = get_ngrams(train_0.values.flatten(), 1, ngram_dict = {})
    counts_1 = get_ngrams(train_1.values.flatten(), 1, ngram_dict = {})
    # this gets all tokens in our training sets to count them
    tokens = list(train_0.values.flatten()) + list(train_1.values.flatten())
    counts = get_ngrams(tokens, 1, ngram_dict = {})
    # gets the length of the volcabulary of the training set
    V = len(counts)

    # a dictionary with the conditional probabilities for each word
    cond_prob_0 = calc_cond_prob(dict(counts), dict(counts_0), c_0, V)
    cond_prob_1 = calc_cond_prob(dict(counts), dict(counts_1), c_1, V)

    # prior class probabilites
    prob_0 = (len(train_0)/(len(train_0) + len(train_1)))
    prob_1 = (len(train_1)/(len(train_0) + len(train_1)))

    # # # Step 4: Testing our classifier
    pred = get_classification(test, cond_prob_0, cond_prob_1, prob_0, prob_1, pair)
    test['target'] = pred['pred']

    confusion = confusion_matrix(truth['target'], test['target'])
    confusion = pd.DataFrame(confusion, columns = pair, index = pair)
    accuracy = accuracy_score(truth['target'], test['target'])

    f = open("bayes_results.txt", "a+")
    f.write(str(confusion))
    f.write("\n")
    f.write(str(accuracy))
    f.write("\n")
    f.close()

# fin #
