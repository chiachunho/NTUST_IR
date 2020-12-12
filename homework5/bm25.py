# %%
import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
from numba import jit
import scipy.sparse
from collections import Counter
import pickle

# %%
load_from_file = True

# %%
# load doc list
with open('doc_list.txt') as f:
    doc_list = f.read().splitlines()


# %%
# load doc from list
if load_from_file != True:
    docs_counter = []
    words = set()
    for doc in tqdm(doc_list):
        with open('docs/' + doc + '.txt') as f:
            doc_words = f.read().split()
            docs_counter.append(Counter(doc_words))
            # words = words.union(set(doc_words))
    with open('docs_counter.pickle', 'wb') as f:
        pickle.dump(docs_counter, f, pickle.HIGHEST_PROTOCOL)
else:
    with open('docs_counter.pickle', 'rb') as f:
        docs_counter = pickle.load(f)

# %%
# load query list
with open('query_list.txt') as f:
    query_list = f.read().splitlines()


# %%
# load query from list
queries = []
queries_words = set()
for query in tqdm(query_list):
    with open('queries/' + query + '.txt') as f:
        query_words = f.read().split()
        queries.append(query_words)
        if load_from_file == False:
            words = words.union(set(query_words))
            queries_words = queries_words.union(set(query_words))

if load_from_file:
    # load query words from file
    with open('query_word_list.txt') as f:
        queries_words = f.read().split()
else:
    # save query words
    with open('query_word_list.txt', 'w') as f:
        f.write(' '.join(queries_words))
    
    queries_words = list(queries_words)


# %%
if load_from_file:
    # load words dict from file
    with open('bm25_word_list.txt') as f:
        words = f.read().split()
else:
    # save words
    with open('bm25_word_list.txt', 'w') as f:
        f.write(' '.join(words))

    words = list(words)

# %%
docs_amount = len(doc_list)
words_amount = len(words)
query_word_amount = len(queries_words)

print(docs_amount, words_amount, query_word_amount)

# %%
# word index dict

words_index = {}

for word_index in tqdm(range(words_amount)):
    words_index[words[word_index]] = word_index


# %%
# term frequency in document

if load_from_file:
    tf_docs = np.load('bm25_tf_docs.npy')

else:
    tf_docs = np.empty((docs_amount, words_amount))

    for doc_index in tqdm(range(docs_amount)):
        for word_index in range(words_amount):
            tf_docs[doc_index][word_index] = docs_counter[doc_index][words[word_index]]

    np.save('bm25_tf_docs', tf_docs)

# %%
# document frequency

if load_from_file:
    words_df = np.load('bm25_words_df.npy')

else:
    words_df = np.empty(words_amount)

    for word_index in tqdm(range(words_amount)):
        words_df[word_index] = np.count_nonzero(tf_docs[:, word_index])

    np.save('bm25_words_df', words_df)

# %%
# inverse document frequency

if load_from_file:
    words_idf = np.load('bm25_words_idf.npy')

else:
    words_idf = np.log((docs_amount - words_df + 0.5) / (words_df + 0.5))

    np.save('bm25_words_idf', words_idf)

# %%
# document length

if load_from_file:
    docs_len = np.load('docs_len.npy')
else:
    docs_len = np.empty(docs_amount)

    for doc_index in tqdm(range(docs_amount)):
        docs_len[doc_index] = sum(docs_counter[doc_index].values())

    np.save('docs_len', docs_len)

# %%
# average document length

avg_doclen = np.average(docs_len)

# %%
print(tf_docs.shape, words_df.shape, words_idf.shape, docs_len.shape, avg_doclen)

# %%
# BM25 calculate parameter

K1 = 0.8
b = 0.7

# %%
# generate BM25 weight

queries_result = np.empty((docs_amount, len(query_list)))

for doc_index in tqdm(range(docs_amount)):
    tf_cd = 1 - b + b * (docs_len[doc_index] / avg_doclen)
    for query_index in range(len(query_list)):
        bm25_weight = 0
        for word in queries[query_index]:
            word_index = words_index[word]
            tf_prime = tf_docs[doc_index][word_index] / tf_cd
            bm25_weight += (K1 + 1) * tf_prime / (K1 + tf_prime) * words_idf[word_index]
        queries_result[doc_index][query_index] = bm25_weight

# %% 
# # sort and export result
sim_df = pd.DataFrame(queries_result)
sim_df.index = doc_list
sim_df.columns = query_list
sim_df


# %%
# save results

now = datetime.datetime.now()

save_filename = 'results/bm25_result' + '_' + now.strftime("%y%m%d_%H%M") + '.txt'

print(save_filename)

with open(save_filename, 'w') as f:
    f.write('Query,RetrievedDocuments\n')
    for query in query_list:
        f.write(query + ",")
        query_sim_df = sim_df[query].sort_values(ascending=False)
        f.write(' '.join(query_sim_df[:5000].index.to_list()) + '\n')
