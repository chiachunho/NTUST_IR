# Homework 5 - Query Modeling
B10615043 四資工四甲 何嘉峻

## 使用的 tool
Python, numpy, pandas, collections.Counter, scipy.sparse, numba.jit, datetime

## 資料前處理

1. 將 `doc_list.txt`, `query_list.txt` 讀檔進來後，之後將每個 doc 和 query 使用 `collections.Counter` 儲存。
2. 生成 document 和 query 的 tf-idf
3. Lexicon 生成方式：使用 df 範圍（5 ~ 10000）的單字，過濾一些 stop word 和稀少的單字。
4. 在 `c(w,d)`, `c(w,q)` 使用 sublinear_tf，並先計算好 document 和 query 的 unigram language model，也先計算好 background language model。

## 作業流程

使用 vsm 做為第一次檢索的結果去做 rocchio，再將 rocchio 最好的結果作為 smm relevant document，下去做 smm


## Rocchio 模型參數調整 (homework5_rocchio.py)

* $TF:\begin{cases}
 0 & \text{ if } tf=0 \\ 
 1 + log(tf)& \text{ if } tf>=1 
\end{cases} \ \ IDF: log(\frac{1 + N}{1 + df_{i}}) + 1$
* $Rocchio: \vec{q} = \alpha \cdot \vec{q} + \beta \cdot \frac{1}{\left | R_{q} \right |} \cdot \left ( \sum_{d_{j}\in R_{q}} \vec{d_{j}} \right ) - \gamma \cdot \frac{1}{\left | \bar{R_{q}} \right |} \cdot \left ( \sum_{d_{{j}'}\in \bar{R_{q}}} \vec{d_{{j}'}} \right )$

* 最終使用參數（Kaggle Public Score: 0.54248）：
    * `alpha` = `1` / `beta` = `0.5` / `gamma` = `0.15` / `rele_doc` = `5` / `nrele_doc` = `1` / `epoch` = `5`  

## Simple Mixture Model 參數調整 (homework5_smm.py)

* $KL(q||d_{j}) \propto - \sum_{w\in V} P(w|q)logP(w|d_{j})$
* $P(w|q) = \alpha \times P_{ULM}(w|q) + \beta \times P_{SMM}(w|q) + (1 - \alpha -\beta ) \times P_{BG}(w)$
* $P(w|d_{j}) = \gamma \times P_{ULM}(w|d_{j}) + (1 - \gamma ) \times P_{BG}(w)
$

* 最終使用參數（Kaggle Public Score: 0.59390）：
    * `alpha` = `0.1` / `beta` = `0.85` / `gamma` = `0.2` / `rele_doc` = `5` / `epoch` = `50` / `smm_alpha` = `0.6`  

## 各種 model 的分數表現 
| Model Name   | Parameter                                 | Kaggle Public MAP@5000 | Note                      |
| ------------ | ---------------------------------------------------- | ----------: | ------------------------- |
| VSM (TF-IDF) | smooth_idf, sublinear_tf                             | 0.41864     | 取全部的單字                |
| VSM (TF-IDF) | smooth_idf, sublinear_tf                             | 0.42399     | 只取 query 出現過的單字     |
| BM25         | K1=0.8, K3=1000, b=0.7                               | 0.48936     |                           |
| Rocchio      | a=1, b=0.5, g=0.15, reledoc=5, nreledoc=1, epoch=5   | 0.52744     | 取全部的單字                |
| Rocchio      | a=1, b=0.5, g=0.15, reledoc=5, nreledoc=1, epoch=5   | 0.54248     | 取 df >= 5 的單字          |
| SMM          | smm_a=0.3, reledoc=5, epoch=50, a=0.15, b=0.8, g=0.3 | 0.57964     | 取 df >= 5 的單字          |
| SMM          | smm_a=0.6, reledoc=5, epoch=50, a=0.1, b=0.85, g=0.2 | **0.59390** | 取 10000 <= df >= 5 的單字 |

## 模型運作原理

query model 主要是要解決 query 的資訊量過少的問題，像 rocchio 就是將 query 的向量加上了相關的文件向量也去除了一些些不相關文章向量，讓 query vector 能夠涵蓋更多資訊，SMM 則是透過 training P_smm 和 P_bg 讓更多特定的單字（在 background model 中沒有辦法被找到的單字）能夠在 P_smm 中的機率提高，提升 query model 的判別力。


## 個人心得

　　這次作業一開始寫的時候原本以為會很簡單，想說直接挑戰實作 SMM，結果馬上被單字量嚇到，一開始也做不出來，所以就轉戰 rochhio，但是發現用第一次作業的 VSM 都沒有過 baseline，後來想說先用 sklearn 的 tfidfvectorizer 跑跑看，發現有 sublinear_tf 這個參數可以調整用來縮減 document 長度的問題，發現可以過 baseline，就趕快把公式套上去，這次作業也重做的第一次作業的 VSM，上次作業知道 sparce matrix 後，發現在 VSM 實作效率超級高，後來嘗試去掉一些出現次數較少的單字也有提升一些分數，沒嘗試太多組參數就開始做 SMM，因為這次 VSM 有重做的關係，在前處理建 unigram 和 background model 的時候都比上次 PLSA 的效率快多啦，後面就開始調整 SMM 的參數，然後到了一個瓶頸後又把目前最高分的結果當作 relevant feedback，的確也有再進步，但後面要再用同一個方法的時候就往下掉了一些，調參數的過程中也好擔心 private 的成績會掉超多的 QQ。


<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>
<script type="text/x-mathjax-config"> MathJax.Hub.Config({ tex2jax: {inlineMath: [['$', '$']]}, messageStyle: "none" }); </script>