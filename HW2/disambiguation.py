#!/usr/bin/env python
# encoding: utf-8

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# disambiguation.py: This script disambiguates a few word pairs using
#   either a Naive Bayes classifier or [EDIT] classifier.
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
    return words
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
            train_file = '_'.join([my_word, "training.txt"])
            test_file = '_'.join([my_word, "testing.txt"])
            # get context
            context = get_context(my_word, words, n)
            # train/test split
            msk = np.random.rand(len(context)) < 0.8
            train = np.array(context)[msk]
            test = np.array(context)[~msk]
            # save train/test sets
            save_to_file(train_file, train)
            save_to_file(test_file, test)






# # # # # # # # # # # # # # # #
### Main Script Starts Here ###
# # # # # # # # # # # # # # # #

# # # Clean the corpus
print('Beginning preprocessing...')
start = time.time()
words = clean_corpus(raw)
end = time.time()
print((end - start), 'seconds elapsed')

# # # Step 1: Extract and save wordpairs

# indicate wordpairs of interest here
wordpairs = [("night", "seat"),
             ("kitchen", "cough"),
             ("car", "bike"),
             ("manufacturer", "bike"),
             ("big", "small"),
             ("huge", "heavy")]

# extract and save (only call this once...)
wordpair_context(wordpairs, words, 10)

for pair in wordpairs:
    # # #
    # Step 2: Prep out train/test datasets
    # # #
    train = pd.DataFrame()
    test = pd.DataFrame()
    truth = pd.DataFrame() # a copy of the test dataset with the true target
    for word in pair:
        # declare filenames to grab context
        train_file = '_'.join([word, "training.txt"])
        test_file = '_'.join([word, "testing.txt"])
        # get context
        new_train = pd.read_csv(train_file, delimiter=' ', header=None)
        new_test = pd.read_csv(test_file, delimiter=' ', header=None)
        # append true target value
        new_train["target"] = word
        new_test["target"] = word
        # append dataframes to train/truth sets
        train = train.append(new_train)
        truth = truth.append(new_test)
    # shuffle our dataframes once both wordpairs are present
    train = train.sample(frac=1)
    truth = truth.sample(frac=1)
    # drop our known target for our test set
    test = truth.drop(["target"], axis=1)

    # # #
    # Step 3: Training our classifiers. On top our standard naive bayes I'm
    #   going to try out some wild card ML classifiers. Because I expect these
    #   to take longer, I'll only test them out on a few instances of my
    #   word pairs.
    # # #
    wildcard_wordpairs = [("car", "bike"),
                          ("big", "small"),
                          ("huge", "heavy")]


    # While unecessary, I will save the classifer output to a txt for
    # curiosities sake... It might be neat to see *where* my classifiers
    # performed bad/good.


# # # Bonus Section:

# Here I'll repeat the process but with smaller/larger windows

# Here I'll experiment with a balanced training set



# fin #
