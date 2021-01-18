# Homework 6 - Transformer-based models
B10615043 四資工四甲 何嘉峻

## 使用的 tool
Python, numpy, pandas, ml_metrics.mapk, Pytorch, transformers, datetime, re

## 資料前處理

1. 將 `documents.csv` 去除 類似 tag 的東西，像是 `<tag> </tag>`，另外如果有明確標注有 `[Text]` 的文章，將文章內容從 `[Text]` 後開始擷取。
2. 分割訓練：驗證集 8 : 2
3. 每個 query 取 1:3 positive:negative (from BM25-top1000) 的文章，將 input 整理成 `[CLS] query [SEP] document [SEP]`
4. Dataloader 每筆 instance 會給 `input_ids` , `attention_mask` , `token_type_ids` , `label` 

## 作業流程

1. 使用 `BertForMultipleChoice` 進行訓練，參數使用助教提供的 baseline 超參數
2. 訓練出模型後使用測試集計算出純 BERT 的 MAP@1000 分數
3. 找出最佳的 MAP@1000 的 $\alpha$ for $score_{final} = score_{BM25} + \alpha \times score_{BERT}$
4. 實際測試 Kaggle 上的 test_queries 並使用剛剛的找出的 $\alpha$ 綜合 BM25+BERT 的 MAP 成績


## 模型參數 (by TA's baseline setting)

```python
# Input limitation
max_query_length = 64
max_input_length = 512
num_negatives = 3   # num. of negative documents to pair with a positive document

# Model finetuning
model_name_or_path = "bert-base-uncased"
max_epochs = 2 # edited by me
learning_rate = 3e-5
dev_set_ratio = 0.2   # make a ratio of training set as development set for rescoring weight sniffing
max_patience = 0      # earlystop if avg. loss on development set doesn't decrease for num. of epochs
batch_size = 2    # num. of inputs = 8 requires ~9200 MB VRAM (num. of inputs = batch_size * (num_negatives + 1))
num_workers = 2   # num. of jobs for pytorch dataloader
```


## 個人心得

　　這次作業要不是助教的挖空 baseline notebook 我一定做不出來 QQ。

​		上課的時候覺得自己都聽得懂沒什麼太大的問題，但在實作的時候真的完全沒有頭緒，雖然知道需要 dataloader 那些東西，但是實際要怎麼實作因為沒有經驗的關係，還是有點茫然，一開始真的想說乾脆不要做作業 6 了 QQ，但是後來看到助教都這麼佛心了，不寫真的對不起自己也對不起助教的用心，原本有想說要把範例的 notebook 試著換成 RoBERTa 來做做看，結果好像 input 的地方有需要調整，沒辦法直接替換的樣子，找了很久還是不太清楚發生什麼問題 zz，所以後來就調整 dataset 的 data 而已，另外試試看 epoch 調成 2 看 training 的效果會不會比較好。 

## Reference
* https://huggingface.co/transformers/model_doc/bert.html#transformers.BertForMultipleChoice
* https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html#a-gentle-introduction-to-torch-autograd
* https://leemeng.tw/attack_on_bert_transfer_learning_in_nlp.html


<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>
<script type="text/x-mathjax-config"> MathJax.Hub.Config({ tex2jax: {inlineMath: [['$', '$']]}, messageStyle: "none" }); </script>