# %%
import time
import datetime
import numpy as np
import pandas as pd
import scipy.sparse
from tqdm import tqdm
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
def counters_conv2_tf_idf(counter_list, vocabulary_dict, smooth_idf=True, sublinear_tf=False):
    # term frequency
    data = []
    col = []
    row = []
    for index in tqdm(range(len(counter_list)), desc="TF"):
        doc_words = list(counter_list[index])
        for word in doc_words:
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

tf_idf_docs = scipy.sparse.csr_matrix.multiply(docs_tf, idf).tocsr()
tf_idf_queries = scipy.sparse.csr_matrix.multiply(queries_tf, idf).tocsr()

first_round_result = cosine_similarity(tf_idf_docs, tf_idf_queries)

# %%
sim_df = pd.DataFrame(first_round_result)
sim_df.index = doc_list
sim_df.columns = query_list


# %%
sim_df
# %%
# save results
now = datetime.datetime.now()
save_filename = 'results/vsm_result' + '_word'+ str(len(words)) + now.strftime("_%y%m%d_%H%M") + '.txt'
print(save_filename)

with open(save_filename, 'w') as f:
    f.write('Query,RetrievedDocuments\n')
    for query in query_list:
        f.write(query + ",")
        query_sim_df = sim_df[query].sort_values(ascending=False)
        f.write(' '.join(query_sim_df[:5000].index.to_list()) + '\n')
# %%

def export_to_csv(result, word, alpha, reledocs, method='rocchio', epoch=1):

    # result dataframe
    sim_df = pd.DataFrame(result)
    sim_df.index = doc_list
    sim_df.columns = query_list

    # save results
    now = datetime.datetime.now()
    save_filename = 'results/'+ method +'_result' + '_word'+ str(word) + '_a' + str(alpha) + '_reledocs' + str(reledocs) + '_epoch' + str(epoch) + now.strftime("_%y%m%d_%H%M") + '.txt'
    print(save_filename)

    with open(save_filename, 'w') as f:
        f.write('Query,RetrievedDocuments\n')
        for query in query_list:
            f.write(query + ",")
            query_sim_df = sim_df[query].sort_values(ascending=False)
            f.write(' '.join(query_sim_df[:5000].index.to_list()) + '\n')

# %%
alpha = 0.3
relevant_docs_amount = 5
EPOCH = 5

# %%

prev_round_result = cosine_similarity(tf_idf_docs, tf_idf_queries)
prev_tf_idf_queries = tf_idf_queries


for iter in tqdm(range(EPOCH)):
    re_tf_idf_queries = scipy.sparse.csr_matrix(tf_idf_queries.shape, dtype=np.float64)

    for q in range(len(query_list)):
        fixed_rq_doc_vector = scipy.sparse.csr_matrix((1, len(words)), dtype=np.float64)
        for doc_index in np.flip(np.argsort(prev_round_result[:,q]))[:relevant_docs_amount]:
            fixed_rq_doc_vector += tf_idf_docs[doc_index]
        fixed_rq_doc_vector /= relevant_docs_amount
        
        re_tf_idf_queries[q] = alpha * tf_idf_queries[q] + (1 - alpha) * fixed_rq_doc_vector

    # Calculate second-round TF-IDF(doc, reformulate-query) Vector Space Model
    prev_round_result = cosine_similarity(tf_idf_docs, re_tf_idf_queries)
    prev_tf_idf_queries = re_tf_idf_queries

    export_to_csv(prev_round_result, len(words), alpha, relevant_docs_amount, epoch=iter+1)

# %%
