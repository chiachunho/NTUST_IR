# %%
import datetime
import numpy as np
import pandas as pd
import scipy.sparse
from tqdm import tqdm
from numba import njit, jit
from collections import Counter

# %%
# load doc list
with open('doc_list.txt') as f:
    doc_list = f.read().splitlines()

doc_index = {}
for i, doc in enumerate(doc_list):
    doc_index[doc] = i

# load query list
with open('query_list.txt') as f:
    query_list = f.read().splitlines()

# load query-relevant doc list
with open('results/smm_result_word34933_a1_b0.5_g0.15_reledocs5_nreledocs0_epoch40_smm_a0.3_201216_2157.txt') as f:
    line = f.readline()
    relevant_list = []
    for line in f.readlines():
        rele_list = line.split(',')[1].split()
        relevant_list.append([doc_index[doc] for doc in rele_list[:5]])
        
# %%
def create_dict_counters(doc_list, query_list):
    # words = set()
    # queries_words = set()
    docs_counter = []
    queries_counter = []
    
    for doc in tqdm(doc_list, desc="Documents"):
        with open('docs/' + doc + '.txt') as f:
            doc_counter = Counter(f.read().split())
            docs_counter.append(doc_counter)
            # words = words.union(set(doc_counter))

    for query in tqdm(query_list, desc="Queries"):
        with open('queries/' + query + '.txt') as f:
            query_counter = Counter(f.read().split())
            queries_counter.append(query_counter)
            # words = words.union(set(query_counter))
            # queries_words = queries_words.union(set(query_counter))
    
    # words = list(words)
    # queries_words = list(queries_words)

    return docs_counter, queries_counter
# %%

docs_counter, queries_counter = create_dict_counters(doc_list=doc_list, query_list=query_list)

# %%
def counters_conv2_tf_idf(counter_list, vocabulary=None, smooth_idf=True, sublinear_tf=False, token_len=1):
    vocabulary_dict = dict()
    if vocabulary != None:
        # preproccess_words_index 
        new_word_index = 0
        for word in tqdm(vocabulary, desc="Vocabulary"):
            if len(word) >= token_len:
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
                col.append(vocabulary_dict.setdefault(word, len(vocabulary_dict)))
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

    return tf, idf, list(vocabulary_dict)

# %%

docs_tf, idf, words = counters_conv2_tf_idf(counter_list=docs_counter, smooth_idf=True, sublinear_tf=True, token_len=2)
queries_tf, query_idf, _ = counters_conv2_tf_idf(counter_list=queries_counter, vocabulary=words, smooth_idf=True, sublinear_tf=True, token_len=2)

# %%

def df_filter(tf_list, idf, vocabulary, samples_n, min_df=1, max_df=None, smooth_idf=True, keep_word=None):
    if smooth_idf:
        min_idf_filter = np.log((1 + samples_n) / (1 + min_df)) + 1
    else:
        min_idf_filter = np.log(samples_n / min_df) + 1
    
    idf_filter = idf <= min_idf_filter

    if max_df != None:
        if max_df > 0 and max_df <= 1:
            max_df = samples_n * max_df
        if max_df > 1:
            if smooth_idf:
                max_idf_filter = np.log((1 + samples_n) / (1 + max_df)) + 1
            else:
                max_idf_filter = np.log(samples_n / max_df) + 1

        idf_filter = np.logical_and(idf_filter, idf >= max_idf_filter)

    if type(keep_word) != type(None):
        if smooth_idf:
            keep_idf_filter = np.log((1 + keep_word[0]) / (1 + 1)) + 1
        else:
            keep_idf_filter = np.log(keep_word[0] / 1) + 1
            
        idf_filter = np.logical_or(idf_filter, keep_word[1] <= keep_idf_filter)

    slim_words_index = np.where(idf_filter)[0]

    for i in range(len(tf_list)):
        tf_list[i] = tf_list[i][:, slim_words_index]
    idf = idf[slim_words_index]

    vocabulary = [vocabulary[index] for index in slim_words_index]

    return tf_list, idf, vocabulary

# %%
[docs_tf, queries_tf], idf, words = df_filter([docs_tf, queries_tf], idf, words, len(doc_list), min_df=5, max_df=0.33)

# %%
@njit
def _unigram_model_data(data, row, indices, indptr):
    for r_index in range(row):
        r_sum = 0.0
        for c_index in range(indptr[r_index], indptr[r_index + 1]):
            r_sum += data[c_index]
        # prevent csr bug
        if r_sum != 0:
            for c_index in range(indptr[r_index], indptr[r_index + 1]):
                data[c_index] /= r_sum
    return data

def unigram_model(tf):
    tf = tf.copy()
    tf.data = _unigram_model_data(tf.data, tf.shape[0], tf.indices, tf.indptr)
    return tf
# %%
doc_unigram = unigram_model(docs_tf).T
query_unigram = unigram_model(queries_tf).T 

# %%
def bg_model(data, row, indices, indptr):
    bg = np.zeros(row)
    for r_index in range(row):
        r_sum = 0.0
        for c_index in range(indptr[r_index], indptr[r_index + 1]):
            r_sum += data[c_index]
        bg[r_index] = r_sum
    bg /= bg.sum()
    return np.array(bg)

# %%
# transpose matrix
cwd = docs_tf.copy().tocoo().transpose().tocsr()
cwq = queries_tf.copy().tocoo().transpose().tocsr()

# %%
# background model
bg = bg_model(cwd.data, cwd.shape[0], cwd.indices, cwd.indptr)

# %%
# smm

smm_alpha = 0.6
EPOCH = 50

# %%
# initial 
psmm = np.random.random_sample(size=queries_tf.shape)
for i in range(psmm.shape[0]):
    psmm[i] /= psmm[i].sum()

# %%
for _ in tqdm(range(EPOCH)):
    # E-step
    ptsmm = np.divide((1 - smm_alpha) * psmm, (1 - smm_alpha) * psmm + smm_alpha * bg)

    # M-Step
    for q_index in range(len(query_list)):
        q_cwd_sum = cwd[:,relevant_list[q_index]].toarray().sum(axis=1)
        psmm[q_index] = ptsmm[q_index] * q_cwd_sum
        psmm[q_index] /= psmm[q_index].sum()
# %%
# KL-Div parameter
alpha = 0.1
beta = 0.85
gamma = 0.2

# %%

pwq = alpha * query_unigram.T + beta * psmm + (1 - alpha - beta) * bg
# %%
pwd = gamma * doc_unigram.T.tocoo().tocsr() + (1 - gamma) * bg
pwd = pwd.T
# %%
KL = pwq * np.log(pwd)
# %%
KL.shape
# %%
KL_ranking = np.flip(KL.argsort(axis=1).A, axis=1)

# %%
def export_to_csv(ranking, method, word, alpha=0, beta=0, gamma=0, reledocs=0, nreledocs=0, epoch=0, comment=None):

    # save results
    now = datetime.datetime.now()
    if comment is None:
        save_filename = f'results/{method}_result_word{word}_a{alpha}_b{beta}_g{gamma}_reledocs{reledocs}_nreledocs{nreledocs}_epoch{epoch}_{now.strftime("%y%m%d_%H%M")}.txt'
    else:
        save_filename = f'results/{method}_result_word{word}_a{alpha}_b{beta}_g{gamma}_reledocs{reledocs}_nreledocs{nreledocs}_epoch{epoch}_{comment}_{now.strftime("%y%m%d_%H%M")}.txt'
    print(save_filename)

    with open(save_filename, 'w') as f:
        f.write('Query,RetrievedDocuments\n')
        for query_index in range(len(query_list)):
            f.write(query_list[query_index] + ",")
            doc_ranking = [doc_list[doc_index] for doc_index in ranking[query_index][:5000]]
            f.write(' '.join(doc_ranking) + '\n')
# %%
export_to_csv(KL_ranking, 'smm', len(words), alpha=alpha, beta=beta, gamma=gamma, reledocs=5, epoch=EPOCH, comment=f'smm_a{smm_alpha}_mindf0.33')

# %%
