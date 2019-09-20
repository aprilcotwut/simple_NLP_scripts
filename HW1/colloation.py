# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# collocation.py:
#
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

import math
import time
import nltk
from nltk.corpus import stopwords as stopwords

stop_words = set(stopwords.words('english'))

file = open('amazon_reviews.txt', 'r', encoding='utf-8')
raw = file.read()p

#
# clean_corpus: minimally preprocesses corpus per nltk parser given by S. Gauch
#   corpus: raw text to parse (str)
def clean_corpus(corpus):
    # just in case, this checks all chars are ascii
    corpus = "".join(char for char in corpus if ord(char)<128)
    # tokenize into words
    tokens = nltk.word_tokenize(corpus)
    # make all chars lowercase and remove non-alpha chars
    words = [word.lower() for word in tokens if word.isalpha()]
    return words

#
# get_ngrams: creates hashtable of ngrams (frequency table for n = 1)
#   words: list containing words to consider (list)
#   n: n value for n-gram (int)
#   dict: optional existing hashtable to append to (dict)
#
def get_ngrams(words, n, ngram_dict = {}):
    for i in range(0, (len(words)-(n-1))):
        new_ngram = ' '.join(words[i:i+n])
        if new_ngram in ngram_dict.keys():
            ngram_dict[new_ngram] += 1
        else:
            ngram_dict[new_ngram] = 1
    ngram_dict = sorted(ngram_dict.items(), key=lambda x:-x[1])
    return ngram_dict

#
# calc_pmi: calculates the pmi given a bigram and frequency hashtable
#   freq_dict: dictionary of word frequencies (dict)
#   bigram_dict: dictionary of bigram frequencies (dict)
#   n: total number of words in the corpus
def calc_pmi(bigram_dict, freq_dict, n):
    keys = [*bigram_dict]
    for key in keys:
        tokens = nltk.word_tokenize(key)
        # pmi = log(prob(w1,w2)/(prob(w1)*prob(w2)))
        # pmi math further discussed in report
        bigram_dict[key] = bigram_dict[key]/(freq_dict[tokens[0]]*freq_dict[tokens[1]])
        bigram_dict[key] = math.log(n*bigram_dict[key])
    bigram_dict = sorted(bigram_dict.items(), key=lambda x:-x[1])
    return bigram_dict





#
# filter_bigrams: exploratory function to reduce filter invalid bigrams
#   bigram_dict: an existing dictionary of bigrams (dict)
def filter_bigrams(bigram_dict):
    first_pos = ('JJ', 'JJR', 'JJS', 'NN', 'NNS', 'NNP', 'NNPS')
    second_pos = ('NN', 'NNS', 'NNP', 'NNPS')
    keys = [*bigram_dict]
    for key in keys:
        delete = False
        tokens = nltk.word_tokenize(key)
        pos_tags = nltk.pos_tag(tokens)
        if not (pos_tags[0][1] in first_pos and pos_tags[1][1] in second_pos):
            delete = True
        for word in tokens:
            if word in stop_words:
                delete = True
        if delete:
            del bigram_dict[key]
    bigram_dict = sorted(bigram_dict.items(), key=lambda x:-x[1])
    return bigram_dict



# # #
# Let's first try finding the bigrams without any additional processing or
# word frequency analysis...
# # #
words = clean_corpus(raw)
frequency = get_ngrams(words, 1)
for word in words:
    if (frequency[word] == 1)
    words.remove(word)

bigrams = get_ngrams(words, 2)
print("These are the top 100 bigrams in the corpus:")
for i in range(0, 100):
    print(bigrams[i])

# # #
# Now we'll use Pointwise Mutual Information to determine the "true" top 100
# collocations
# # #
frequency = get_ngrams(words, 1)
bigram_pmi = calc_pmi(dict(bigrams), dict(frequency), len(words))

# # #
# Now let's try developing out bigrams in a more intellegent way. We will
# instead read in the corpus in phrase by phrase by chopping up our string where
# there is a comma or period. Additionally stop words will be removed, and only
# bigrams of the form "Noun Noun" and "Adjective Noun" will be accepted. This
# will probably take longer, but I'm curious
# how it performs...
# # #
start = time.time()
filtered = filter_bigrams(dict(bigrams))
end = time.time()
print((end - start), 'seconds elapsed')

print("These are the top 100 bigrams in the corpus post-filter:")
for i in range(0, 100):
    print(bigrams[i])

# # #
# Now we'll use Pointwise Mutual Information to determine the "true" top 100
# collocations of the filtered bigrams
# # #
