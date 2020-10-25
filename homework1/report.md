# Homework 1 - Vector Space Model
B10615043 四資工四甲 何嘉峻

## 使用的 tool
Python, Jupyter, numpy, dataframe, sklearn.metrics.pairwise.cosine_similarity, datetime

## 資料前處理

一開始我將 `doc_list` 和  `query_list` 讀檔進來後，之後將每個 doc 和 query 都使用 `split()` 儲存起來，並將這些 list 放到 `set` 中製作 dictionary，這邊做了一個小偷吃步，直接把 dictionary 的範圍縮小到只看所有 `query` 出現過的字，把 dictionary 的維度從 59680 降到了 123 而已。

## 模型參數調整

我的 document 和 query 的 term weight 都是使用此公式：

$$tf_{i,j} \times log(1+\frac{N+1}{n_{i}+1})$$


## 模型運作原理

vector space model 會給出一個 doc 或 query 對應到所有 dictionary 中的向量，所以我們使用一個 doc 的向量和 query 的向量去做 cosine similarity，我們就可以得到兩者間的相似程度，所以我們將所有 doc 的向量都跟 query 的向量算過相似度後，我們就可以從高排到低找出跟該 query 最相關的 doc。

## 個人心得

一開始模仿 sklearn 套件的公式，算出來的結果剛好超過 baseline 一些些，但大家的分數都蠻高的，所以請教別人後才知道有只看 query 出現過的字這個偷吃步的方法，除了讓程式效率變高之外，也讓我的分數往上不少。

<script type="text/javascript"
  src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML">
</script>