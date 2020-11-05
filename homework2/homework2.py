import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm

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

# save words
with open('min_word_list.txt', 'w') as f:
    f.write(' '.join(words))
    
words = list(words)

# load words dict from file
with open('min_word_list.txt') as f:
    words = f.read().split()

## Calculate docment-tf, query-tf, df, idf

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
    idf.append(np.log((docs_len - df + 0.5) / (df + 0.5)))
idf_npy = np.array(idf)
np.save('min_idf_npy', idf_npy)

## Load calculated matrix (save calculate time)
tf_docs_npy = np.load('min_tf_docs_npy.npy')
tf_queries_npy = np.load('min_tf_queries_npy.npy')
df_npy = np.load('min_df_npy.npy')
idf_npy = np.load('min_idf_npy.npy')

## BM25 calculate

K1 = 0.28
K3 = 1000
b = 0.85

avg_doclen = 0
for doc in docs.values():
    avg_doclen += len(doc)
avg_doclen /= len(docs)

queries_result = []

for query_id, query in tqdm(queries.items()):
    query_result = []
    query_index = query_list.index(query_id)
    for doc_name, doc_content in docs.items():
        bm25_weight = 0
        doc_index = doc_list.index(doc_name)
        doc_len = len(doc_content)
        for word in query:
            word_index = words.index(word)
            tf_ij = tf_docs_npy[doc_index][word_index]
            tf_iq = tf_queries_npy[query_index][word_index]
            idf_i = idf_npy[word_index]
            single_term_weight = idf_i * (K1 + 1) * tf_ij / (tf_ij + K1 * ((1 - b) + b * doc_len / avg_doclen)) # * (K3 + 1) * tf_iq / (K3 + tf_iq)
            bm25_weight += single_term_weight
        query_result.append(bm25_weight)
    queries_result.append(query_result)

## sort and export result
sim_df = pd.DataFrame(queries_result)
sim_df = sim_df.transpose()
sim_df.index = doc_list
sim_df.columns = query_list

# save results
now = datetime.datetime.now()
save_filename = 'results/result' + '_' + now.strftime("%y%m%d_%H%M") + '.txt'
print(save_filename)

with open(save_filename, 'w') as f:
    f.write('Query,RetrievedDocuments\n')
    for query in query_list:
        f.write(query + ",")
        query_sim_df = sim_df[query].sort_values(ascending=False)
        f.write(' '.join(query_sim_df.index.to_list()) + '\n')


