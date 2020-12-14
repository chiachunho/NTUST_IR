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

words_index = dict()

for word_index in range(len(words)):
    words_index[words[word_index]] = word_index
# %%
def preproccess_words(vocabulary):
    new_word_index = 0
    index_dict = dict()
    result_words = []
    for word in tqdm(vocabulary, desc="Vocabulary"):
        if len(word) > 1:
            result_words.append(word)
            index_dict[word] = new_word_index
            new_word_index += 1
    
    return result_words, index_dict
# %%

words, words_index = preproccess_words(words)
# %%
def counters_conv2_tf_idf(counter_list, vocabulary_dict, smooth_idf=True, sublinear_tf=False):
    # term frequency
    data = []
    col = []
    row = []
    for index in tqdm(range(len(counter_list)), desc="TF"):
        doc_words = list(counter_list[index])
        for word in doc_words:
            if len(word) > 1:
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

    return tf, idf
# %%

docs_tf, idf = counters_conv2_tf_idf(counter_list=docs_counter, vocabulary_dict=words_index, smooth_idf=True, sublinear_tf=True)
docs_tf.shape
# %%

queries_tf, _ = counters_conv2_tf_idf(counter_list=queries_counter, vocabulary_dict=words_index, smooth_idf=True, sublinear_tf=True)
queries_tf

# %%

def l2_norm(matrix):
    # for r_index in tqdm(range(matrix.shape[0])):
    #     matrix[r_index] /= np.dot(matrix[r_index], matrix[r_index].T).power(0.5).data[0]
    # return matrix
    for r_index in tqdm(range(matrix.shape[0])):
        matrix[r_index] /= np.sqrt(np.sum(np.power(matrix[r_index].data, 2)))
    return matrix
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
# export_to_csv(retrieval_ranking, "svm", len(words))

# %%
alpha = 1
beta = 0.75
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

    export_to_csv(retrieval_ranking, 'rocchio', len(words), alpha=alpha, beta=beta, reledocs=reledocs_amount, nreledocs=n_reledocs_amount, epoch=iter+1)

# %%

# for iter in tqdm(range(EPOCH)):
#     re_tf_idf_queries = scipy.sparse.csr_matrix(tf_idf_queries.shape, dtype=np.float64)

#     for q in range(len(query_list)):
#         fixed_rq_doc_vector = scipy.sparse.csr_matrix((1, len(words)), dtype=np.float64)
#         for doc_index in retrieval_ranking[q][:reledocs_amount]:
#             fixed_rq_doc_vector += tf_idf_docs[doc_index]
#         fixed_rq_doc_vector = fixed_rq_doc_vector.mean(axis=0)
        
#         re_tf_idf_queries[q] = alpha * tf_idf_queries[q] + beta * fixed_rq_doc_vector

#     # Calculate second-round TF-IDF(doc, reformulate-query) Vector Space Model
#     tf_idf_queries = re_tf_idf_queries
#     retrieval_result = cosine_similarity(tf_idf_docs, tf_idf_queries)
#     retrieval_ranking = np.flip(retrieval_result.argsort(axis=0), axis=0).T

#     export_to_csv(retrieval_ranking, 'rocchio', len(words), alpha=alpha, beta=beta, reledocs=reledocs_amount, epoch=iter+1)

# %%
