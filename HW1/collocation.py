# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# collocation.py: This script finds collocations in the given amazon
#   reviews corpus. First, purely frequency is used as a measure then
#   Pointwise Mutual Information is used. Additionally, a more exploratory
#   portion of this assignment attempts additional preprocessing to compare
#   results
#
# Author: April Walker
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
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
# calc_pmi: calculates the pmi given a bigram and frequency hashtable
#   input--
#   freq_dict: dictionary of word frequencies (dict)
#   bigram_dict: dictionary of bigram frequencies (dict)
#   n: total number of words in the corpus
#
#   output--
#   bigram_dict: dictionary of bigrams with pmi (dict)
def calc_pmi(bigram_dict, freq_dict, n):
    keys = [*bigram_dict]
    for key in keys:
        tokens = nltk.word_tokenize(key)
        # pmi = log(prob(w1,w2)/(prob(w1)*prob(w2)))
        # pmi math further discussed in report
        bigram_dict[key] = bigram_dict[key]/(freq_dict[tokens[0]]*freq_dict[tokens[1]])
        bigram_dict[key] = math.log(n*bigram_dict[key])
    # return as is
    return bigram_dict


#
# filter_bigrams: exploratory function to reduce filter invalid bigrams
#   input--
#   bigram_dict: an existing dictionary of bigrams (dict)
#
#   output--
#   bigram_dict: the filtered dictionary of bigrams as list (list)
def filter_bigrams(bigram_dict):
    # let first word be adj or noun, second noun only
    first_pos = ('JJ', 'JJR', 'JJS', 'NN', 'NNS', 'NNP', 'NNPS')
    second_pos = ('NN', 'NNS', 'NNP', 'NNPS')
    keys = [*bigram_dict]
    for key in keys:
        delete = False
        tokens = nltk.word_tokenize(key)
        pos_tags = nltk.pos_tag(tokens)
        # if invalid part of speach tags, delete bigram
        if not (pos_tags[0][1] in first_pos and pos_tags[1][1] in second_pos):
            delete = True
        # if bigram contains stopword, delete
        for word in tokens:
            if word in stop_words:
                delete = True
        if delete:
            del bigram_dict[key]
    bigram_dict = sorted(bigram_dict.items(), key=lambda x:-x[1])
    return bigram_dict

#
# run_test: runs various tests on the functions in this script
#   input--
#   words: an array of strings conatining the words in the corpus (list)
def run_test(words):
    file = open('test.log', 'w')
    # test that clean_corpus removes all non-alpha chars, makes all chars
    #   lowercase, and removes words only occuring once
    text = 'cat Cat CAT dog doG meow 300 300'
    file.write('String before preprocessing: \n')
    file.write(text)
    file.write('\n')
    file.write('Array after preprocessing: \n')
    text = clean_corpus(text)
    file.write(str(text))
    file.write('\n')
    # test that get_ngram will not get incomplete ngrams at the end of the corpus
    #   (that is a less than 2 word "bigram")
    file.write('Testing get_ngrams()...')
    bigrams = get_ngrams(words, 2)
    keys = [*dict(bigrams)]
    test_passed = True
    for k in keys:
        tokens = nltk.word_tokenize(k)
        if not (len(tokens) == 2):
            test_passed = False
    if test_passed:
        file.write('Test passed!')
    else:
        file.write('Test failed!')
    file.close()



# # # Clean the corpus
print('Beginning preprocessing...')
start = time.time()
words = clean_corpus(raw)
end = time.time()
print((end - start), 'seconds elapsed')

# # # Test
run_test(words[0:100])

# # #
# Let's first try finding the bigrams without any additional processing or
# word frequency analysis...
# # #
print('Finding bigrams...')
start = time.time()
bigrams = get_ngrams(words, 2)
end = time.time()
print((end - start), 'seconds elapsed')

print('These are the top 100 bigrams in the corpus using frequency:')
for i in range(0, 100):
    print(bigrams[i])

# # #
# Now we'll use Pointwise Mutual Information to determine the "true" top 100
# collocations
# # #
frequency = get_ngrams(words, 1)

print('Finding PMI of bigrams...')
start = time.time()
bigram_pmi = calc_pmi(dict(bigrams), dict(frequency), len(words))
end = time.time()
print((end - start), 'seconds elapsed')

pmi_list = list(bigram_pmi.items())
print("These are the PMI's of the top 100 bigrams:")
for i in range(0, 100):
    print(pmi_list[i])

bigram_pmi = sorted(bigram_pmi.items(), key=lambda x:-x[1])
print("These are the bigrams with the top 100 highest PMI's:")
for i in range(0, 100):
    print(bigram_pmi[i])

# # #
# Now let's try developing out bigrams in a more intellegent way. Stop words
# will be removed, and only bigrams of the form "Noun Noun" and "Adjective Noun"
# will be accepted. This will probably take longer, but I'm curious how it
# performs...
# # #
print('Exploratory additional preprocessing comence...')
start = time.time()
filtered = filter_bigrams(dict(bigrams))
end = time.time()
print((end - start), 'seconds elapsed')

print('These are the top 100 bigrams in the corpus post-filter:')
for i in range(0, 100):
    print(filtered[i])

# # #
# Now we'll use Pointwise Mutual Information to determine the "true" top 100
# collocations of the filtered bigrams
# # #
filtered_pmi = calc_pmi(dict(filtered), dict(frequency), len(words))
pmi_list = list(filtered_pmi.items())
print("These are the PMI's of the top 100 bigrams:")
for i in range(0, 100):
    print(pmi_list[i])

filtered_pmi = sorted(filtered_pmi.items(), key=lambda x:-x[1])
print("These are the bigrams with the top 100 highest PMI's:")
for i in range(0, 100):
    print(filtered_pmi[i])

# fin #
