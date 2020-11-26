# %%
import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
from numba import jit
import scipy.sparse
from collections import Counter


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
# document length
if load_from_file:
    docs_len = np.load('docs_len.npy')
else:
    docs_len = []

    for j in tqdm(range(docs_amount)):
        docs_len.append(sum(docs_counter[j].values()))

    docs_len = np.array(docs_len)
    np.save('docs_len', docs_len)


# %%
# all words count in documents and probability
if load_from_file:
    cwd = scipy.sparse.load_npz('cwd.npz')
    pwd = scipy.sparse.load_npz('pwd.npz')
else:
    indptr = [0]
    indices = []
    cwd_data = []
    pwd_data = []

    for j in tqdm(range(docs_amount)):
        doc_len = docs_len[j]

        for i in range(words_amount):
            word_count = docs_counter[j][words[i]]
            if word_count != 0:
                indices.append(i)
                cwd_data.append(word_count)
                pwd_data.append(word_count / doc_len)
        indptr.append(len(indices))

    cwd = scipy.sparse.csr_matrix((cwd_data, indices, indptr), dtype=np.float32).transpose()
    pwd = scipy.sparse.csr_matrix((pwd_data, indices, indptr), dtype=np.float32).transpose()

    scipy.sparse.save_npz('cwd', cwd)
    scipy.sparse.save_npz('pwd', pwd)


# %%
# process slim words
if load_from_file:
    with open('slim_word_list_10000.txt') as f:
        slim_words = f.read().split()
else:
    words_count_list = []
    for word_row in tqdm(cwd):
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
# slim words count in documents and probability
if load_from_file:
    slim_cwd = scipy.sparse.load_npz('slim_cwd_10000.npz').A
    slim_pwd = scipy.sparse.load_npz('slim_pwd_10000.npz').A
else:
    indptr = [0]
    indices = []
    cwd_data = []
    pwd_data = []

    for j in tqdm(range(docs_amount)):
        doc_len = docs_len[j]
        for i in range(slim_words_amount):
            word_count = docs_counter[j][slim_words[i]]
            if word_count != 0:
                indices.append(i)
                cwd_data.append(word_count)
                pwd_data.append(word_count / doc_len)
        indptr.append(len(indices))

    slim_cwd = scipy.sparse.csr_matrix((cwd_data, indices, indptr), dtype=np.float32).transpose()
    slim_pwd = scipy.sparse.csr_matrix((pwd_data, indices, indptr), dtype=np.float32).transpose()

    scipy.sparse.save_npz('slim_cwd_10000', slim_cwd)
    scipy.sparse.save_npz('slim_pwd_10000', slim_pwd)

    slim_cwd = slim_cwd.A
    slim_pwd = slim_pwd.A


# %%
# background language model
bg = []
bg_model_cd = docs_len.sum()

for word_row in tqdm(slim_cwd):
    bg.append(word_row.sum() / bg_model_cd)

bg = np.array(bg)


# %%
@jit(nopython=True)
def nb_E_step(pwt, ptd, cwd, topic_amount, word_amount, doc_amount):
    # empty matrix
    ptwd = np.empty((topic_amount, word_amount, doc_amount))

    # Common Denominator
    # ptwd_CD = np.dot(pwt, ptd) 

    for i in range(word_amount):
        for j in range(doc_amount):
            if cwd[i][j] != 0: 
                ptwd_CD = 0
                for k in range(topic_amount):
                    single_ptwd = pwt[i][k] * ptd[k][j]
                    ptwd[k][i][j] = single_ptwd
                    ptwd_CD += single_ptwd
                if ptwd_CD != 0:
                    for k in range(topic_amount):
                        ptwd[k][i][j] /= ptwd_CD
                else:
                    ptwd[:,i,j] = 0
            else:
                ptwd[:,i,j] = 0
    return ptwd


# %%
@jit(nopython=True)
def nb_M_step(ptwd, cwd, docs_len, topic_amount, word_amount, doc_amount):
    # empty matrix
    pwt = np.empty((word_amount, topic_amount))
    ptd = np.empty((topic_amount, doc_amount))

    for k in range(topic_amount):
        single_topic_wd = np.multiply(cwd, ptwd[k])

        # p(w/t)
        single_wt_sum = single_topic_wd.sum()
        if single_wt_sum != 0:
            for i in range(word_amount):
                pwt[i][k] = single_topic_wd[i].sum() / single_wt_sum
        else:
            pwt[:,k] = 1 / slim_words_amount

        # p(t/d)
        for j in range(doc_amount):
            ptd[k][j] = single_topic_wd[:,j].sum() / docs_len[j]
            # ptd[k][j] = single_topic_wd[:,j].sum() / cwd[:,j].sum()
    
    # # norm to 1
    # for k in range(topic_k):
    #     if np.isnan(pwt[:,k].sum()):
    #         print(times, "norm ", k)
    #     pwt[:,k] /= pwt[:,k].sum()
    # for j in range(doc_amount):
    #     deno = ptd[:,j].sum()
    #     if deno != 0:
    #         ptd[:,j] /= deno
    #     else:
    #         ptd[:,j].fill(1 / topic_amount) 
    
    return pwt, ptd


# %%
@jit(nopython=True)
def nb_loss(times, cwd, pwt, ptd):
    loss = np.multiply(cwd, np.log(np.dot(pwt, ptd))).sum()
    print("\nStep", times, "loss: ", loss)


# %%
# topic

topic_k = 48
EPOCH = 30
alpha = 0.7
beta = 0.25


# %%
# EM Step Initial (normal)
pwt = np.random.random(size = (slim_words_amount, topic_k))

for k in range(topic_k):
    pwt[:,k] /= pwt[:,k].sum()

ptd = np.full((topic_k, docs_amount), 1 / topic_k)

ptwd = np.empty((topic_k, slim_words_amount, docs_amount))


# %%
for i in tqdm(range(EPOCH)):
    # nb_E_step(pwt, ptd, cwd, topic_amount, word_amount, doc_amount)
    ptwd = nb_E_step(pwt, ptd, slim_cwd, topic_k, slim_words_amount, docs_amount)
    # nb_M_step(ptwd, cwd, docs_len, topic_amount, word_amount, doc_amount)
    pwt, ptd = nb_M_step(ptwd, slim_cwd, docs_len, topic_k, slim_words_amount, docs_amount)
    # nb_loss(times, cwd, pwt, ptd)
    # nb_loss(i + 1, slim_cwd, pwt, ptd)


# %%
plsa_EM_final = np.matmul(pwt, ptd)


# %%
plsa_EM_final.shape


# %%
queries_result = []

for query in tqdm(queries):
    query_result = []
    for doc_index in range(docs_amount):
        plsa_result = 1
        for word in query:
            word_index = slim_words.index(word)
            unigram_pwd = slim_pwd[word_index][doc_index]
            plsa_result = plsa_result * (alpha * unigram_pwd + beta * plsa_EM_final[word_index][doc_index] + (1 - alpha - beta) * bg[word_index])
            
            # plsa_result = plsa_result * (alpha * unigram_pwd + (1 - alpha - beta) * bg[word_index])
        query_result.append(plsa_result)
    queries_result.append(query_result)


# %%
## sort and export result
sim_df = pd.DataFrame(queries_result)
sim_df = sim_df.transpose()
sim_df.index = doc_list
sim_df.columns = query_list


# %%
sim_df


# %%
# save results
now = datetime.datetime.now()
save_filename = 'results/result' + '_' + 'topic' + str(topic_k) + '_EPOCH' + str(EPOCH) + '_a' + str(alpha) + '_b' + str(beta) + '_word'+ str(slim_words_amount) + now.strftime("_%y%m%d_%H%M") + '.txt'
print(save_filename)

with open(save_filename, 'w') as f:
    f.write('Query,RetrievedDocuments\n')
    for query in query_list:
        f.write(query + ",")
        query_sim_df = sim_df[query].sort_values(ascending=False)
        f.write(' '.join(query_sim_df[:1000].index.to_list()) + '\n')


# %%
sparse_pwt = scipy.sparse.csr_matrix(pwt)
sparse_ptd = scipy.sparse.csr_matrix(ptd)


# %%
scipy.sparse.save_npz('sparse_pwt' + '_' + 'topic' + str(topic_k) + '_EPOCH' + str(EPOCH) + '_word'+ str(slim_words_amount) + now.strftime("_%y%m%d_%H%M"), sparse_pwt)
scipy.sparse.save_npz('sparse_ptd' + '_' + 'topic' + str(topic_k) + '_EPOCH' + str(EPOCH) + '_word'+ str(slim_words_amount) + now.strftime("_%y%m%d_%H%M"), sparse_ptd)


# %%
pwt = scipy.sparse.load_npz('sparse_pwt_topic48_EPOCH30_word10020_201126_0310.npz').A
ptd = scipy.sparse.load_npz('sparse_ptd_topic48_EPOCH30_word10020_201126_0310.npz').A


# %%
for i in range(100):
    print(ptd[:,i].sum())


# %%
for i in range(topic_k):
    print(pwt[:,i].sum())


# %%
for i in range(topic_k):
    topic_top10_words_index = np.flip(np.argsort(pwt[:,i]))[:10]
    topic_top10_words = [slim_words[index] for index in topic_top10_words_index]
    print('Topic ', i, ': ', ' '.join(topic_top10_words))


# %%



