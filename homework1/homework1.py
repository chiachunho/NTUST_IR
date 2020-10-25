#!/usr/bin/env python
# coding: utf-8

import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

# load doc list
with open('doc_list.txt') as f:
    doc_list = f.read().splitlines()

# load doc from list
docs = {}
words = set()
for doc in tqdm(doc_list):
    with open('docs/' + doc + '.txt') as f:
        data = f.read()
        docs[doc] = data.split()
#         words = words.union(set(docs[doc]))

# load query list
with open('query_list.txt') as f:
    query_list = f.read().splitlines()

# load query from list
queries = {}
for query in tqdm(query_list):
    with open('queries/' + query + '.txt') as f:
        data = f.read()
        queries[query] = data.split()
        words = words.union(set(queries[query]))


# # Dictionary save / load

# save words
with open('min_word_list.txt', 'w') as f:
    f.write(' '.join(words))
    
words = list(words)

# load words dict from file
with open('min_word_list.txt') as f:
    words = f.read().split()


# # Calculate docment-tf, query-tf, df, idf

# term frequency in document
tf_docs_list = []

for content in tqdm(docs.values()):
    tf_doc = []
    for word in words:
        tf_doc.append(content.count(word))
    tf_docs_list.append(tf_doc)

tf_docs_npy = np.array(tf_docs_list)
np.save('min_tf_docs_npy', tf_docs_npy)


# document frequency
df_list = []

for word in tqdm(words):
    count = 0
    for content in docs.values():
        if word in content:
            count += 1
    df_list.append(count)

df_npy = np.array(df_list)
np.save('min_df_npy', df_npy)


# term frequency in query
tf_queries_list = []

for content in tqdm(queries.values()):
    tf_query = []
    for word in words:
        tf_query.append(content.count(word))
    tf_queries_list.append(tf_query)

tf_queries_npy = np.array(tf_queries_list)
np.save('min_tf_queries_npy', tf_queries_npy)


# inverse document frequency
idf = []
docs_len = len(docs)

for df in tqdm(df_npy):
    idf.append(np.log(1 + (1 + docs_len) / (1 + df)))

idf_npy = np.array(idf)
np.save('min_idf_npy', idf_npy)


# # Load calculated matrix (save calculate time)

tf_docs_npy = np.load('min_tf_docs_npy.npy')
tf_queries_npy = np.load('min_tf_queries_npy.npy')
df_npy = np.load('min_df_npy.npy')
idf_npy = np.load('min_idf_npy.npy')

# # Calculate TF-IDF(doc, query)
# term weight (tf i,j)*log(1 + (1+N)/(1+ni))

tf_idf_docs = tf_docs_npy * idf_npy
tf_idf_queries = tf_queries_npy * idf_npy

# # cosine similarity (doc, query)
cosine_npy = cosine_similarity(tf_idf_docs, tf_idf_queries)

# # sort and export result
sim_df = pd.DataFrame(cosine_npy)
sim_df.index = doc_list
sim_df.columns = query_list

now = datetime.datetime.now()
save_filename = 'results/result' + '_' + now.strftime("%y%m%d_%H%M") + '.txt'

print(save_filename)

with open(save_filename, 'w') as f:
    f.write('Query,RetrievedDocuments\n')
    for query in query_list:
        f.write(query + ",")
        query_sim_df = sim_df[query].sort_values(ascending=False)
        f.write(' '.join(query_sim_df.index.to_list()) + '\n')

