# %%
import time
import datetime
import numpy as np
import pandas as pd
import scipy.sparse
from tqdm import tqdm
from numba import njit
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
# %%
# load doc list
with open('doc_list.txt') as f:
    doc_list = f.read().splitlines()

# load query list
with open('query_list.txt') as f:
    query_list = f.read().splitlines()
# %%

def create_dict_counters(doc_list, query_list):
    words = set()
    queries_words = set()
    docs_counter = []
    queries_counter = []
    
    for doc in tqdm(doc_list, desc="Documents"):
        with open('docs/' + doc + '.txt') as f:
            doc_counter = Counter(f.read().split())
            docs_counter.append(doc_counter)
            words = words.union(set(doc_counter))

    for query in tqdm(query_list, desc="Queries"):
        with open('queries/' + query + '.txt') as f:
            query_counter = Counter(f.read().split())
            queries_counter.append(query_counter)
            words = words.union(set(query_counter))
            queries_words = queries_words.union(set(query_counter))
    
    words = list(words)
    queries_words = list(queries_words)

    return words, queries_words, docs_counter, queries_counter
# %%

words, queries_words, docs_counter, queries_counter = create_dict_counters(doc_list=doc_list, query_list=query_list)

# %%
def counters_conv2_tf_idf(counter_list, vocabulary, smooth_idf=True, sublinear_tf=False, token_len=1):
    # preproccess_words_index 
    new_word_index = 0
    vocabulary_dict = dict()
    result_words = []
    for word in tqdm(vocabulary, desc="Vocabulary"):
        if len(word) >= token_len:
            result_words.append(word)
            vocabulary_dict[word] = new_word_index
            new_word_index += 1

    # term frequency
    data = []
    col = []
    row = []
    for index in tqdm(range(len(counter_list)), desc="TF"):
        doc_words = list(counter_list[index])
        for word in doc_words:
            if len(word) >= token_len:
                data.append(counter_list[index][word])
                col.append(vocabulary_dict[word])
                row.append(index)
    if sublinear_tf:
        data = 1 + np.log(data)
    tf = scipy.sparse.csr_matrix((data, (row, col)), shape=(len(counter_list), len(vocabulary_dict)), dtype=np.float64)

    # document frequency
    df_counter = Counter(col)
    df = []
    for i in tqdm(range(len(vocabulary_dict)), desc="DF"):
        df.append(df_counter[i])
    df = np.array(df)

    # inverse document frequency
    if smooth_idf:
        # idf(t) = log [ (1 + n) / (1 + df(t)) ] + 1
        idf = np.log((1 + len(counter_list)) / (1 + df)) + 1
    else:
        # idf(t) = log [ n / df(t) ] + 1
        idf = np.log(len(counter_list) / df) + 1

    return tf, idf, result_words
# %%

docs_tf, idf, words = counters_conv2_tf_idf(counter_list=docs_counter, vocabulary=words, smooth_idf=True, sublinear_tf=True, token_len=2)
queries_tf, _, _ = counters_conv2_tf_idf(counter_list=queries_counter, vocabulary=words, smooth_idf=True, sublinear_tf=True, token_len=2)

# %%

def min_df_filter(tf_list, idf, min_df, vocabulary, samples_n, smooth_idf=True):
    if smooth_idf:
        idf_filter = np.log((1 + samples_n) / (1 + min_df)) + 1
    else:
        idf_filter = np.log(samples_n / min_df) + 1
    
    slim_words_index = np.where(idf <= idf_filter)[0]
    for i in range(len(tf_list)):
        tf_list[i] = tf_list[i][:, slim_words_index]
    idf = idf[slim_words_index]

    vocabulary = [vocabulary[index] for index in slim_words_index]

    return tf_list, idf, vocabulary

# %%
[docs_tf, queries_tf], idf, words = min_df_filter([docs_tf, queries_tf], idf, 5, words, len(doc_list))
# %%
@njit
def _l2_norm(data, row, indices, indptr):
    power_data = np.square(data)
    for r_index in range(row):
        r_sum = 0.0
        for c_index in range(indptr[r_index], indptr[r_index + 1]):
            r_sum += power_data[c_index]
        # prevent csr bug
        if r_sum != 0:
            r_sum = np.sqrt(r_sum)
            for c_index in range(indptr[r_index], indptr[r_index + 1]):
                data[c_index] /= r_sum
    return data

# %%
def l2_norm(matrix):
    data = matrix.data
    shape = matrix.shape
    indices = matrix.indices
    indptr = matrix.indptr
    dtype = matrix.dtype
    l2_norm_data = _l2_norm(data, shape[0], indices, indptr)
    l2_norm_matrix = scipy.sparse.csr_matrix((l2_norm_data, indices, indptr), shape=shape, dtype=dtype)
    
    return l2_norm_matrix
# %%

tf_idf_docs = scipy.sparse.csr_matrix.multiply(docs_tf, idf).tocsr()
tf_idf_queries = scipy.sparse.csr_matrix.multiply(queries_tf, idf).tocsr()

# l2 norm
tf_idf_docs = l2_norm(tf_idf_docs)
tf_idf_queries = l2_norm(tf_idf_queries)

retrieval_result = cosine_similarity(tf_idf_docs, tf_idf_queries)
retrieval_ranking = np.flip(retrieval_result.argsort(axis=0), axis=0).T

# %%
def export_to_csv(ranking, method, word, alpha=0, beta=0, gamma=0, reledocs=0, nreledocs=0, epoch=0):

    # save results
    now = datetime.datetime.now()
    save_filename = 'results/'+ method +'_result' + '_word'+ str(word) + '_a' + str(alpha) + '_reledocs' + str(reledocs) + '_epoch' + str(epoch) + now.strftime("_%y%m%d_%H%M") + '.txt'
    save_filename = f'results/{method}_result_word{word}_a{alpha}_b{beta}_g{gamma}_reledocs{reledocs}_nreledocs{nreledocs}_epoch{epoch}_{now.strftime("%y%m%d_%H%M")}.txt'
    print(save_filename)

    with open(save_filename, 'w') as f:
        f.write('Query,RetrievedDocuments\n')
        for query_index in range(len(query_list)):
            f.write(query_list[query_index] + ",")
            doc_ranking = [doc_list[doc_index] for doc_index in ranking[query_index][:5000]]
            f.write(' '.join(doc_ranking) + '\n')

# %%
# save results
# export_to_csv(retrieval_ranking, "vsm", len(words))

# %%
alpha = 1
beta = 0.5
gamma = 0.15
reledocs_amount = 5
n_reledocs_amount = 1
EPOCH = 10

# %%
# Calculate second or more-round TF-IDF(doc, reformulate-query) Vector Space Model

for iter in tqdm(range(EPOCH)):
    for q in range(len(query_list)):
        reledocs_vector = tf_idf_docs[retrieval_ranking[q][:reledocs_amount]]
        reledocs_vector = scipy.sparse.csr_matrix(reledocs_vector.mean(axis=0))
        n_reledocs_vector = tf_idf_docs[retrieval_ranking[q][-n_reledocs_amount:]]
        n_reledocs_vector = scipy.sparse.csr_matrix(n_reledocs_vector.mean(axis=0))
        tf_idf_queries[q] = alpha * tf_idf_queries[q] + beta * reledocs_vector - gamma * n_reledocs_vector

    # Calculate second-round TF-IDF(doc, reformulate-query) Vector Space Model
    retrieval_result = cosine_similarity(tf_idf_docs, tf_idf_queries)
    retrieval_ranking = np.flip(retrieval_result.argsort(axis=0), axis=0).T

    export_to_csv(retrieval_ranking, 'rocchio', len(words), alpha=alpha, beta=beta, gamma=gamma, reledocs=reledocs_amount, nreledocs=n_reledocs_amount, epoch=iter+1)

# %%
