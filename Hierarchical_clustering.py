##########Hierarchical Clustering SR Symptom###########
import numpy as np
import pandas as pd
import nltk
import re
import os
import codecs
from sklearn import feature_extraction
import mpld3
import pandas as pd
from gensim import corpora
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from ast import literal_eval
import gensim
from gensim.models import Phrases
from nltk.tokenize import RegexpTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from pprint import pprint
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
import spacy
# Plotting tools
import pyLDAvis
import pyLDAvis.gensim
import matplotlib.pyplot as plt
from nltk.corpus import stopwords 
import re

def unique_words(sentence, number):
    return [w for w in set(sentence.translate(None, punctuation).lower().split()) if len(w) >= number]

def preprocessing(line):
    line = line.lower()
    line = re.sub(r"[{}]".format(string.punctuation), " ", line)
    return line

df = pd.read_csv("All_SRNotes_Extracted.csv", encoding='utf-8')

#If groupby not done...use below code
df = df.groupby('sr_number').agg({'sr_number':'first', 'Keywords': ', '.join})
keywords = df['Keywords'].values.tolist()
for i in range(len(keywords)):
    keywords[i] = keywords[i].replace('[], ', '')
    keywords[i] = keywords[i].replace('], [', ',')

'''
#Else
keywords = df['Keywords'].values.tolist()
'''

docs = []
for i in range(len(keywords)):
    s = ''
    for keyword in literal_eval(keywords[i])[:3]:
        #print(keyword)
        keyword = keyword.split(':')[0]
        keyword = ' '.join([i for i in keyword.split(' ') if not i.isdigit()])
        keyword = ' '.join([i for i in keyword.split(' ') if '/' not in i])
        keyword = re.sub("\S*\d\S*", "", keyword).strip()
        for w in keyword:
            if((ord(w) < 65 or ord(w) > 124) and (ord(w) is not 32) and (ord(w) is not 95)):
                keyword = keyword.replace(w, '')
        #s = s + (' '.join(set(keyword.split()))) + ','
        s = s + keyword + ','
    docs.append(s.rstrip(','))

# Split the documents into tokens.
tokenizer = RegexpTokenizer(r'\w+')
for idx in range(len(docs)):
    docs[idx] = docs[idx].lower()  # Convert to lowercase.
    docs[idx] = tokenizer.tokenize(docs[idx])  # Split into words.

# Remove numbers, but not words that contain numbers.
docs = [[token for token in doc if not token.isnumeric()] for doc in docs]

# Remove words that are only one character.
docs = [[token for token in doc if len(token) > 1] for doc in docs]

# Lemmatize all words in documents.
lemmatizer = WordNetLemmatizer()
docs = [[lemmatizer.lemmatize(token) for token in doc] for doc in docs]


# Add bigrams and trigrams to docs (only ones that appear 5 times or more).
bigram = Phrases(docs, min_count=3)
bigram_mod = gensim.models.phrases.Phraser(bigram)
for idx in range(len(docs)):
    for token in bigram[docs[idx]]:
        if '_' in token:
            # Token is a bigram, add to document.
            if(token not in docs[idx]):
                docs[idx].append(token)

trigram = Phrases(bigram[docs], min_count = 3)#threshold=100
trigram_mod = gensim.models.phrases.Phraser(trigram)
for idx in range(len(docs)):
    for token in trigram[docs[idx]]:
        if '_' in token:
            # Token is a bigram, add to document.
            if(token not in docs[idx]):
                docs[idx].append(token)


stopwords = nltk.corpus.stopwords.words('english')
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")
def tokenize_and_stem(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems


def tokenize_only(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    return filtered_tokens

totalvocab_stemmed = []
totalvocab_tokenized = []
synopses = []
for d in docs:
    synopses.append(' '.join(d))

for i in synopses:
    allwords_stemmed = tokenize_and_stem(i) #for each item in 'synopses', tokenize/stem
    totalvocab_stemmed.extend(allwords_stemmed) #extend the 'totalvocab_stemmed' list    
    allwords_tokenized = tokenize_only(i)
    totalvocab_tokenized.extend(allwords_tokenized)


vocab_frame = pd.DataFrame({'words': totalvocab_tokenized}, index = totalvocab_stemmed)
print('there are ' + str(vocab_frame.shape[0]) + ' items in vocab_frame')

from sklearn.feature_extraction.text import TfidfVectorizer

#define vectorizer parameters
tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=200000,
                                 min_df=0.2, stop_words='english',
                                 use_idf=True, tokenizer=tokenize_and_stem, ngram_range=(1,3))

tfidf_matrix = tfidf_vectorizer.fit_transform(synopses) #fit the vectorizer to synopses

print(tfidf_matrix.shape)
terms = tfidf_vectorizer.get_feature_names()
from sklearn.metrics.pairwise import cosine_similarity
dist = 1 - cosine_similarity(tfidf_matrix)

from scipy.cluster.hierarchy import ward, dendrogram

linkage_matrix = ward(dist) #define the linkage_matrix using ward clustering pre-computed distances

#Number of clusters is from the plots from pervious algorithm
#Threshold for eucledian distance is maximum of these initital cluster distances

import sys
sys.setrecursionlimit(1500)
'''
fig, ax = plt.subplots(figsize=(15, 20)) # set size
ax = dendrogram(linkage_matrix, orientation="right");#, labels=titles);

plt.tick_params(\
    axis= 'x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom='off',      # ticks along the bottom edge are off
    top='off',         # ticks along the top edge are off
    labelbottom='off')

plt.tight_layout() #show plot with tight layout

#uncomment below to save figure
plt.savefig('ward_clusters.png', dpi=200) #save figure as ward_clusters
'''
from sklearn.metrics.pairwise import euclidean_distances
import sys
#def computing_hierarchical_clustering(linkage_matrix, num_clusters):

#linkage_matrix = ward(dist) #define the linkage_matrix using ward clustering pre-computed distances
linkage_matrix = l.copy()
distances = euclidean_distances(linkage_matrix, linkage_matrix)
minimums = []
indexes = []
for i in range(len(distances)):
    d = list(distances[i])
    d[i] = sys.maxsize
    #d = d[:i]+d[i+1:]
    minimums.append(min(d))
    indexes.append(d.index(min(d)))

old_minimums = minimums.copy()

#def get_initial_clusters(minimums, indexes, num_clusters):
num_clusters = 7
clusters = {}
words = {}
similarities = {}
threshold = 0
for i in range(num_clusters):
    minimum = minimums.index(min(minimums))
    clusters[minimum] = [indexes[minimum]]
    similarities[minimum] = [minimums[minimum]]
    words[model.wv.index2word[minimum]] = [model.wv.index2word[indexes[minimum]]]
    threshold = max(threshold, minimums[minimum])
    minimums[minimum] = sys.maxsize
    minimums[indexes[minimum]] = sys.maxsize
    res = (linkage_matrix[minimum] + linkage_matrix[indexes[minimum]])/2
    linkage_matrix[minimum] = res
    linkage_matrix[indexes[minimum]] = res


def get_closer_index(clusters, cluster, linkage_matrix, threshold):
    a = list(euclidean_distances(linkage_matrix, [linkage_matrix[cluster]]))
    a[cluster] = sys.maxsize
    for j in clusters.keys():
        for i in clusters[j]:
            a[i] = sys.maxsize
    for j in clusters.keys():
        a[j] = sys.maxsize
    value = min(a)
    index = list(a).index(min(a))
    #print(value)
u    if(value <= threshold):
        return (value, index)
    else:
        return None

def update_linkage_matrix(cluster, linkage_matrix):
    res = numpy.mean(linkage_matrix[cluster], axis=0)
    for i in cluster:
        linkage_matrix[i] = res

def get_length_dict(clusters):
    total_length = 0
    for k in list(clusters.keys()):
        total_length = total_length + len(clusters[k])
    return total_length


import numpy
#def get_all_clusters(linkage_matrix, num_clusters, clusters):
threshold = 1500
total_length = get_length_dict(clusters)
prev_length = 0
while(total_length <= 3000 and total_length != prev_length):
    #total_length = get_length_dict(clusters)
    print(total_length, prev_length)
    prev_length = total_length
    for cluster in list(clusters.keys()):
        v = get_closer_index(clusters, cluster, linkage_matrix, threshold)
        if(v is not None):
            clusters[cluster].append(v[1])
            similarities[cluster].append(v[0][0])
            words[model.wv.index2word[cluster]].append(model.wv.index2word[v[1]])
            update_linkage_matrix([cluster]+clusters[cluster], linkage_matrix)
    total_length = get_length_dict(clusters)

for k in clusters.keys():
    print(len(clusters[k]))
    print(len(similarities[k]))



'''

sentences = ['hi', 'hello', 'hi hello', 'goodbye', 'bye', 'goodbye bye']
sentences_split = [s.lower().split(' ') for s in sentences]

import gensim
model = gensim.models.Word2Vec(sentences_split, min_count=2)

from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

l = linkage(model.wv.syn0, method='complete', metric='seuclidean')

# calculate full dendrogram
plt.figure(figsize=(25, 10))
plt.title('Hierarchical Clustering Dendrogram')
plt.ylabel('word')
plt.xlabel('distance')

a = dendrogram(
    l,
    leaf_rotation=90.,  # rotates the x axis labels
    leaf_font_size=16.,  # font size for the x axis labels
    orientation='left',
    leaf_label_func=lambda v: str(model.wv.index2word[v])
)
plt.show()

words = a['ivl']
leaves = a['leaves']
colours = a['color_list']
cluster_words = {}
for colour in set(colours):
    cluster_words[colour] = []

for i in range(len(colours)):
    cluster_words[colours[i]].append(words[i])

'''
