# BERT Variants I - ALBERT,RoBERTa,ELECTRA,and SpanBERT
We will start with understanding how ALBERT works. ALBERT is basically A Lite version of BERT model.

## A Lite version of BERT 
ALBERT is a lite version of BERT with fewer parameters compared to BERT. It uses the following two techniques to reduce the number of parameters:
使用了兩種技術去減少網絡參數量 
- Cross-layer parameter sharing 跨層參數共享
- Factorized embedding layer parameterization 分解嵌入層參數化


## Extracting embeddings with ALBERT 
> ALBERT_extracting_embeddings.py
> 
## Robustly Optimized BERT pre-trained 
> RoBERTa_exploring_tokenizer
## Understanding ELECTRA 
ELECTRA (Efficiently Learning an Encoder that Classifies Token Replacements Accurately) is yet another interesting variant of BERT. 

ELECTRA（高效學習一種準確分類令牌替換的編碼器）是 BERT 的另一個有趣變體。
> ELECTRA_training_method.py


## Predicting span with SpanBERT 
SpanBERT is mostly used for tasks such as question answering where we predict the span of text.
SpanBERT 主要用於預測文本跨度的問答等任務。
