#!/usr/bin/env python
# encoding: utf-8

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# got_social_nx.py: This script develops social networks for Game of Thrones
#
# Author: April Walker
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


import math
import time
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns


import nltk
from nltk.corpus import stopwords as stopwords
stop_words = set(stopwords.words('english'))

import sklearn
from sklearn import metrics


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
    return ' '.join(words)

#
# get_vocab: creates hashtable of word frequency
#   input--
#   words: list containing words to consider (list)
#   words_dict: optional existing hashtable to append to (dict)
#   sorted: IDs whether to sort the hashtable before returning (bool)
#
#
#   output--
#   ngram_dict: the sorted dict of ngrams as a list (list)
def get_vocab(words, sort, words_dict = {}):
    for i in range(0, (len(words)-(2))):
        # get new ngram
        new_word = ' '.join(words[i:i+1])
        # if ngram exists in dictionary add occurance
        if new_word in words_dict.keys():
            words_dict[new_word] += 1
        # else add to dictionary
        else:
            words_dict[new_word] = 1
    # if specified, sort decending by occurance and return list
    if sort:
        words_dict = dict(sorted(words_dict.items(), key=lambda x:-x[1]))
    return words_dict

#
# filter_vocab: exploratory function to reduce filter stop words from vocab
#   input--
#  words_dict: an existing dictionary of vocab (dict)
#
#   output--
#   words_dict: the filtered dictionary of bigrams as list (list)
def filter_vocab(words_dict):
    # let first word be adj or noun, second noun only
    keys = [*words_dict]
    for key in keys:
        delete = False
        # if bigram contains stopword, delete
        if key in stop_words:
            del words_dict[key]
    words_dict = dict(sorted(words_dict.items(), key=lambda x:-x[1]))
    return words_dict


# # # # # # # # # # # # # # # #
### Main Script Starts Here ###
# # # # # # # # # # # # # # # #

df = pd.read_csv('got_scripts_breakdown.csv', sep=';')
df = df.drop(columns = ['Column1', 'Season', 'Episode', 'N_serie', 'N_Season',
                        'Emision Date'])

for i in range(0, df.shape[0]):
    df.at[i, 'Sentence'] = clean_corpus(df['Sentence'][i])

# lists top n_top characters by number of lines...
n_top = 10
top_characters = df['Name'].value_counts()[0:n_top].axes[0].tolist()

#
# Cosine similarity segment
#
top_dialogue = {}
vocab = {}
dialogue = ''

# First find cosine similarity for the top 15 characters entire vocab
for name in top_characters:
    top_dialogue[name] = df.loc[df['Name'] == name].drop(columns = 'Name')
    dialogue = dialogue + ' ' + ' '.join(top_dialogue[name].values.flatten().tolist())

dialogue = nltk.word_tokenize(dialogue)
vocab = get_vocab(words = dialogue, sort = True, words_dict = {})

top_vocab = {}

for name in top_characters:
    vocab = dict.fromkeys(vocab, 0)
    this_dialogue = ' '.join(top_dialogue[name].values.flatten().tolist())
    this_dialogue = nltk.word_tokenize(this_dialogue)
    this_vocab = get_vocab(words = this_dialogue, sort = False, words_dict = vocab)
    top_vocab[name] = this_vocab

cos_df = pd.DataFrame(-0.001, columns = top_characters, index = top_characters)

for name1 in top_characters:
    for name2 in top_characters:
        tmp1 = np.array(list(top_vocab[name1].values())).flatten().reshape(1, -1)
        tmp2 = np.array(list(top_vocab[name2].values())).flatten().reshape(1, -1)
        cos_sim = float(sklearn.metrics.pairwise.cosine_similarity(tmp1, tmp2).flatten().round(3))
        cos_df[name1][name2] = cos_sim
        print('The cosine similarity between ' + name1 + ' and ' + name2 + ' is ' + str(cos_sim))

sns.heatmap(cos_df, linewidth=0.5, annot=True)
plt.savefig("cos_sim_fig.png")
plt.show()


# Then find cosine similarity for the top 15 characters filtered vocab
filtered_top_vocab = {}

for name in top_characters:
    filtered_top_vocab[name] = filter_vocab(top_vocab[name])

filtered_cos_df = pd.DataFrame(-0.001, columns = top_characters, index = top_characters)

for name1 in top_characters:
    for name2 in top_characters:
        tmp1 = np.array(list(filtered_top_vocab[name1].values())).flatten().reshape(1, -1)
        tmp2 = np.array(list(filtered_top_vocab[name2].values())).flatten().reshape(1, -1)
        cos_sim = float(sklearn.metrics.pairwise.cosine_similarity(tmp1, tmp2).flatten().round(3))
        filtered_cos_df[name1][name2] = cos_sim
        print('The cosine similarity between ' + name1 + ' and ' + name2 + ' is ' + str(cos_sim))

sns.heatmap(filtered_cos_df, linewidth=0.5, annot=True)
plt.savefig("filtered_cos_sim_fig.png")
plt.show()

# Now develop a list of hashtables containing the number of times a character spoke
# to another character...

# declare nx graph and add nodes
g = nx.Graph()
g.add_nodes_from(top_characters)

pos = nx.spring_layout(g)
nx.draw_networkx_nodes(g,pos,node_color='darksage',node_size=7500)

labels = {}
for node_name in top_characters:
    labels[str(node_name)] = str(node_name)
nx.draw_networkx_labels(g,pos,labels,font_size=16)



# get weights for each character
edge_weights = dict()
for character in top_characters:
    # init list of hashtables
    edge_weights[character] = dict((name, 0) for name in top_characters)

for i in range(1, len(df['Name'])):
    name1 = df['Name'][i]
    name2 = df['Name'][i-1]
    if (name1 in top_characters) and (name2 in top_characters):
        edge_weights[name1][name2] = edge_weights[name1][name2] + 1

for name1 in top_characters:
    for name2 in top_characters:
        if name1 != name2:
            total_weight = edge_weights[name1][name2] + edge_weights[name2][name1]
            if total_weight > 0: g.add_edge(name1, name2, weight = total_weight)

all_weights = []
# iter through graph nodes to gather all the weights
for (node1,node2,data) in g.edges(data=True):
        all_weights.append(data['weight']) #we'll use this when determining edge thickness

unique_weights = list(set(all_weights))

# plot each edge one by one...
for weight in unique_weights:
    # develop filtered list with just the weight you want to draw
    weighted_edges = [(node1,node2) for (node1,node2,edge_attr) in g.edges(data=True) if edge_attr['weight']==weight]
    # multiplying by [num_nodes/sum(all_weights)] makes the graphs edges look cleaner
    width = weight*len(top_characters)*3.0/sum(all_weights)
    nx.draw_networkx_edges(g,pos,edgelist=weighted_edges,width=width)

#Plot the graph
plt.axis('off')
plt.title('Top Character Social Network in Game of Thrones Universe')
plt.savefig("GoT_social_nx_2.png")
plt.show()

g_distance_dict = {(e1, e2): 1 / (weight + 0.1) for e1, e2, weight in g.edges(data='weight')}
nx.set_edge_attributes(g, g_distance_dict, 'distance')

# declare node info dataframe
node_info_cols = ['degree', 'eigen_central', 'close_central']
node_info_df = pd.DataFrame(columns = node_info_cols, index = top_characters)

# save node information to dataframe
for name in top_characters:
    node_info_df['degree'][name] = g.degree(weight='weight')[name]
    node_info_df['close_central'][name] = nx.closeness_centrality(g, distance='distance')[name]
    node_info_df['eigen_central'][name] = nx.eigenvector_centrality(g, weight='weight')[name]
