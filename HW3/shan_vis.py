#!/usr/bin/env python
# encoding: utf-8

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# shan_vis.py: This script builds a language model to generate
#   Shakespearean text using the Shannon Visualization game...
#
# Author: April Walker
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
import numpy as np
import pandas as pd
import ipdb
import re
import math
import time
import nltk
from nltk.corpus import stopwords as stopwords

stop_words = set(stopwords.words('english'))

file = open('Shakespeare.txt', 'r', encoding='utf-8')
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
    # get rid of that informational stuff...
    corpus = re.sub("[\\n]", "", corpus)
    corpus = re.sub("[\<].*?[\>]", "", corpus)
    corpus = re.sub("[\']", "", corpus)
    # keep track of where sentences end wiht identifier
    corpus = re.sub("[\.]", " endsentenceid", corpus)
    # just in case, this checks all chars are ascii
    corpus = ''.join(char for char in corpus if ord(char)<128)
    # tokenize into words
    tokens = nltk.word_tokenize(corpus)
    # make all chars lowercase, remove non-alpha chars
    words = [word.lower() for word in tokens if word.isalpha()]
    return words

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

# calc_cond_prob: calculates the conditional probabilites for all words in dict
#   input--
#   bigrams_dict: dictionary of bigrams counts (dict)
#   unigrams_dict: dictionary of unigrams counts (dict)
#
#   output--
#   bigram_dict: dictionary of "normalized" bigrams
def calc_cond_prob(bigrams_dict, unigrams_dict):
    keys = [*bigrams_dict]
    for key in keys:
        tokens = nltk.word_tokenize(key)
        bigrams_dict[key] = bigrams_dict[key]/unigrams_dict[tokens[0]]
    bigrams_dict = sorted(bigrams_dict.items(), key=lambda x:-x[1])
    return bigrams_dict

# calc_cond_prob: calculates the conditional probabilites for all words in dict
#   input--
#   input_dict: dictionary of bigrams occuring in input sentence (dict)
#
#   output--
#   sent_prob: the probability of the sentence occuring (double)
def calc_sent_prob(input_dict, cond_prob_dict):
    sent_prob = 0
    keys = [*input_dict]
    for key in keys:
        if key in cond_prob_dict:
            sent_prob = sent_prob + math.log(cond_prob_dict[key])
        else:
            # if word doesnt exist use inverse of log(-7) as probability
            sent_prob = sent_prob - 7
    sent_prob = math.exp(sent_prob)
    return(sent_prob)



#
# filter_bigrams: discludes "endsentenceid" containing bigrams for
#   shannon visualization game
#
#   input--
#   bigram_dict: an existing dictionary of bigrams (dict)
#
#   output--
#   bigram_dict: the filtered dictionary of bigrams as list (list)
def filter_bigrams(bigram_dict):
    keys = [*bigram_dict]
    for key in keys:
        delete = False
        tokens = nltk.word_tokenize(key)
        # if bigram contains "endsentenceid", delete
        for word in tokens:
            if word == 'endsentenceid':
                delete = True
        if delete:
            del bigram_dict[key]
    bigram_dict = sorted(bigram_dict.items(), key=lambda x:-x[1])
    return bigram_dict

# run_test: testing testing 123...
def run_test():
    file = open('test.log', 'w')
    file.write('nothing here yet...')
    file.close()




# # # # # # # # # # # # # # # #
### Main Script Starts Here ###
# # # # # # # # # # # # # # # #

run_test()

# # # Step 1: Clean the corpus
print('Beginning preprocessing...')
start = time.time()
words = clean_corpus(raw)
end = time.time()
print((end - start), 'seconds elapsed')

# # # Step 2: Get bigram/unigram hashtable
print('Finding bigrams...')
start = time.time()
bigrams = dict(get_ngrams(words, 2, {}))
end = time.time()
print((end - start), 'seconds elapsed')

print('Finding unigrams...')
start = time.time()
unigrams = dict(get_ngrams(words, 1, {}))
end = time.time()
print((end - start), 'seconds elapsed')

# # # Step 3: Normalize bigram counts
print('Normalizing bigrams...')
start = time.time()
# This includes our "endsentenceid" for determining sentence probability
norm_bigrams = dict(calc_cond_prob(bigrams, unigrams))
# This discludes our id for the shannon visualization game
true_bigrams = filter_bigrams(norm_bigrams)
end = time.time()
print((end - start), 'seconds elapsed')

# uncomment below to see top 100 normalized bigrams
print('These are the top 100 normalized bigrams in the corpus:')
for i in range(0, 100):
    print(true_bigrams[i])

# Step 4: Ask user for sentence to determine the probaility of...
print('Please give me a sentence to determine the probability of...')
sent_input = input()
words_input = clean_corpus(sent_input)
# This id will show that the first word is the beginning of the sentence
words_input.insert(0, "endsentenceid")
if words_input[len(words_input)-1] != "endsentenceid":
    words_input.append("endsentenceid")
bigrams_input = dict(get_ngrams(words_input, 2, {}))
print("The probability is " + str(calc_sent_prob(bigrams_input, norm_bigrams)))

print('Go ahead and try again, but this time use a PHRASE...')
sent_input = input()
words_input = clean_corpus(sent_input)
# This time we'll remove any the end sentence identifier...
for i in range(0, len(words_input)):
    if words_input[i] == "endsentenceid":
        words_input.pop(i)
bigrams_input = dict(get_ngrams(words_input, 2, {}))
print("The probability is " + str(calc_sent_prob(bigrams_input, norm_bigrams)))

# fin #
