# Homework 2 - Best Match Model 25
B10615043 四資工四甲 何嘉峻

## 使用的 tool
Python, Jupyter, numpy, dataframe, datetime

## 資料前處理

1. 將 `doc_list.txt` 和  `query_list.txt` 讀檔進來後，之後將每個 doc 和 query 都使用 `split()` 儲存起來。
2. 跟上次 Vector Space Model 的作業一樣，在生成 Lexicon 時只看 `query_list` 切完的所有詞並放到 `set` 中來生成，這樣能把 Lexicon 的維度從 59680 降到僅 123 而已。
3. 跟上次作業相同先將 document term-frequency, query term-frequency, inverse document frequency 先算好供後面算 BM25 term weight 使用。

## BM25 模型參數調整

* BM25 term weight 公式：

$$sim_{BM25}\left (d_{j}, q \right ) \equiv \sum_{w_{j}\in \left ( d_{j} \cap q \right )}^{} IDF(w_{j}) \times \frac{ \left ( K_{1} + 1 \right ) \times tf_{i,j}}{K_{1} [\left ( 1 - b \right ) + b \times \frac{len\left ( d_{j} \right )}{avg_{doclen}}] + tf_{i,j}} \times \frac{\left ( K_{3} + 1 \right ) \times tf_{i,q}}{ K_{3} + tf_{i,q}}$$

* IDF 公式：

$$IDF(w_{j}) = log\left ( \frac{N - n_{i} + 0.5}{n_{i} + 0.5} \right )$$

* 最終使用參數（Kaggle Public Score: 0.71854）：
    * `K1` = `0.28`
    * `K3` = `1000`
    * `b` = `0.85`
    * `avg_doclen` = `611.3953710331663`
* 使用的參數對照分數表現圖
  ![](https://i.imgur.com/IxbY9i2.png)


## 模型運作原理

BM25 跟 VSM 最大的差異在於多處理了 Document Length Normalization，讓 tf 的重要程度不會直接線性成長，一個字在出現過多次後，已經不會得到更多的分數，一樣的 tf 在較長文章的重要度會低於較短文章。

## 個人心得

在一開始寫作業的時候，算好了 average document length，但是實際在計算 term weight 公式的時候忘記放進去，導致 kaggle 的分數都卡在 0.64 多，後來才發現自己耍蠢忘記除，一開始就先按照簡報上設 b 為 0.75，但都還是沒過 baseline 後，決定上調看看，到 0.8 後就過了 baseline，後來嘗試了調整 K1，發現沒有太大的幅度，K3 項後來我直接捨棄，因為試過幾次沒有含 K3 的分數都一模一樣，這次作業比較晚開始寫，所以也沒有測到很多參數，後來就繼續往上調到 b = 0.85 後，就有比較明顯的進展，雖然跟大家比還是差很多，中間有試過調整看看在算 IDF 取 log 前 +1，但是試過效果都更差，所以後來就還是用簡報上的公式，但後來想想也有可能是沒有找到適合的參數才對，搞不好分數會更高。

<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>
<script type="text/x-mathjax-config"> MathJax.Hub.Config({ tex2jax: {inlineMath: [['$', '$']]}, messageStyle: "none" }); </script>