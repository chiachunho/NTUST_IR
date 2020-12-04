# %%
import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
from numba import jit
import scipy.sparse
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity

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
          words = words.union(set(doc_words))


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
    with open('word_list.txt') as f:
        words = f.read().split()
else:
    # save words
    with open('word_list.txt', 'w') as f:
        f.write(' '.join(words))

    words = list(words)


# %%
docs_amount = len(doc_list)
words_amount = len(words)
query_word_amount = len(queries_words)

print(docs_amount, words_amount, query_word_amount)


# %%
# all words document tf
if load_from_file:
    full_tf = scipy.sparse.load_npz('full_tf.npz')
else:
    indptr = [0]
    indices = []
    tf_data = []

    for j in tqdm(range(docs_amount)):
        for i in range(words_amount):
            word_count = docs_counter[j][words[i]]
            if word_count != 0:
                indices.append(i)
                tf_data.append(word_count)
        indptr.append(len(indices))

    full_tf = scipy.sparse.csr_matrix((tf_data, indices, indptr), dtype=np.float32).transpose()

    scipy.sparse.save_npz('full_tf', full_tf)


# %%
# process slim words
if load_from_file:
    with open('slim_word_list_10000.txt') as f:
        slim_words = f.read().split()
else:
    words_count_list = []
    for word_row in tqdm(full_tf):
        words_count_list.append(word_row.sum())
        
    most_word_index = np.flip(np.argsort(words_count_list))

    slim_words_amount = 10000
    slim_words = []
    for i in range(slim_words_amount):
        slim_words.append(words[most_word_index[i]])
    
    slim_words = slim_words + queries_words
    slim_words = list(set(slim_words))

    slim_words_amount = len(slim_words)

    # save slim words
    with open('slim_word_list_10000.txt', 'w') as f:
        f.write(' '.join(slim_words))

# update slim words amount
slim_words_amount = len(slim_words)
print(slim_words_amount)


# %%
# slim words tf
if load_from_file:
    slim_tf = scipy.sparse.load_npz('slim_tf_10000.npz').A
else:
    indptr = [0]
    indices = []
    tf_data = []

    for j in tqdm(range(docs_amount)):
        for i in range(slim_words_amount):
            word_count = docs_counter[j][slim_words[i]]
            if word_count != 0:
                indices.append(i)
                tf_data.append(word_count)
        indptr.append(len(indices))

    slim_tf = scipy.sparse.csr_matrix((cwd_data, indices, indptr), shape=(docs_amount, slim_words_amount), dtype=np.float32)

    scipy.sparse.save_npz('slim_tf_10000', slim_tf)

    slim_tf = slim_tf.A


# %%
# query tf
if load_from_file:
    query_tf = scipy.sparse.load_npz('query_tf_10000.npz').A
else:
    indptr = [0]
    indices = []
    query_tf_data = []

    for q in tqdm(range(len(query_list))):
        for i in range(slim_words_amount):
            word_count = queries[q].count(slim_words[i])
            if word_count != 0:
                indices.append(i)
                query_tf_data.append(word_count)
        indptr.append(len(indices))

    query_tf = scipy.sparse.csr_matrix((query_tf_data, indices, indptr), shape=(len(query_list), slim_words_amount), dtype=np.float32)

    scipy.sparse.save_npz('query_tf_10000', query_tf)

    query_tf = query_tf.A

# %%
# df and idf

if load_from_file:
    df = scipy.sparse.load_npz('df_10000.npz').A[0]
else:
    df = []
    for word in tqdm(slim_words):
        doc_count = 0
        for j in range(docs_amount):
            if docs_counter[j][word] > 0:
                doc_count += 1
        df.append(doc_count)
    
    df = np.array(df)
    df = scipy.sparse.csr_matrix(df)
    scipy.sparse.save_npz('df_10000', df)
    df = df.A[0]

idf = np.log(1 + (1 + docs_amount) / (1 + df))


# %%
# Calculate first-round TF-IDF(doc, query) Vector Space Model
# term weight: (tf i,j)*log(1 + (1+N)/(1+ni))

tf_idf_docs = np.multiply(slim_tf, idf)
tf_idf_queries = np.multiply(query_tf, idf)

first_round_result = cosine_similarity(tf_idf_docs, tf_idf_queries)

# %%
# reformulate parameter

alpha = 0.5
beta = 0.5
relevant_docs_amount = 100

# %%
# reformulate the query vector by select top-ranked relevant document 

re_tf_idf_queries = np.empty(tf_idf_queries.shape)

for q in tqdm(range(len(query_list))):
    fixed_rq_doc_vector = np.zeros(slim_words_amount)
    for doc_index in np.flip(np.argsort(first_round_result[:,q]))[:relevant_docs_amount]:
        fixed_rq_doc_vector += tf_idf_docs[doc_index]
    fixed_rq_doc_vector /= relevant_docs_amount
    
    re_tf_idf_queries[q] = alpha * tf_idf_queries[q] + beta * re_tf_idf_queries

# %%
# Calculate second-round TF-IDF(doc, reformulate-query) Vector Space Model
second_round_result = cosine_similarity(tf_idf_docs, re_tf_idf_queries)

# %%
# result dataframe
sim_df = pd.DataFrame(second_round_result)
sim_df.index = doc_list
sim_df.columns = query_list


# %%
sim_df


# %%
# save results
now = datetime.datetime.now()
save_filename = 'results/rocchio_result' + '_word'+ str(slim_words_amount) + '_a' + str(alpha) + '_b' + str(beta) + '_reledocs' + str(relevant_docs_amount) + now.strftime("_%y%m%d_%H%M") + '.txt'
print(save_filename)

with open(save_filename, 'w') as f:
    f.write('Query,RetrievedDocuments\n')
    for query in query_list:
        f.write(query + ",")
        query_sim_df = sim_df[query].sort_values(ascending=False)
        f.write(' '.join(query_sim_df[:5000].index.to_list()) + '\n')


# %%



