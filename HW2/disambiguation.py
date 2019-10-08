#!/usr/bin/env python
# encoding: utf-8

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# disambiguation.py: This script disambiguates a few word pairs using
#   either a Naive Bayes classifier or [EDIT] classifier.
#
# Author: April Walker
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
import numpy as np
import math
import time
import nltk
from nltk.corpus import stopwords as stopwords

stop_words = set(stopwords.words('english'))

file = open('amazon_reviews.txt', 'r', encoding='utf-8')
raw = file.read()

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

def wordpair_context(wordpairs, words, n):
    def get_context(my_word, words, n):
        # note word is skipped if appropropriate context not avaliable
        context = []
        for i in range(n, (len(words)-(n-1))):
            if words[i] == my_word:
                # get context for instance of word
                tmp = words[i-n:i+(n+1)]
                tmp.pop(i)
                context.append(' '.join(tmp))
        return context

    for pair in wordpairs:
        for my_word in pair:
            # declare filenames
            train_file = '_'.join(my_word, "training.txt")
            test_file = '_'.join(my_word, "testing.txt")
            # get context
            context = get_context(my_word, words, n)
            # train/test split
            msk = np.random.rand(len(context)) < 0.8
            train = context[msk]
            test = context[~msk]
            # save train/test sets
            fs.writeFile(train_file,
                train.join('\n'),
                function (err) { console.log(''.join('Training data saved for', my_word)); }
            );
            fs.writeFile(test_file,
                test.join('\n'),
                function (err) { console.log(''.join('Testing data saved for', my_word)); }
            );






# # # # # # # # # # # # # # # #
### Main Script Starts Here ###
# # # # # # # # # # # # # # # #

# # # Clean the corpus
print('Beginning preprocessing...')
start = time.time()
words = clean_corpus(raw)
end = time.time()
print((end - start), 'seconds elapsed')

# # # Phase 1: Extract and save wordpairs

# indicate wordpairs of interest here
wordpairs = [("night", "seat"),
             ("kitchen", "cough"),
             ("car", "bike"),
             ("manufacturer", "bike"),
             ("big", "small"),
             ("huge", "heavy")]

# extract and save (only call this once...)
wordpair_extraction(wordpairs, words)

# # # Phase 2: Create pseudowords (truth file creation)

# TODO: for each word, replace word in corpus with pseudoword and create truth
# file of which was the original word. Do at most one replacment per line for
# simplicity... make note of the line number and original word in truth file

# # #  Phase 3: Training our classifiers

# # # Phase 4: Testing our classifiers

# Phase 5 and 6 discussed in paper!

# # # Bonus Section:

# Here I'll repeat the process but with smaller/larger windows

# Here I'll experiment with a balanced training set



# fin #
