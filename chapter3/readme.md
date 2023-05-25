# Getting Hands-on with BERT 
## exploring the pre-trained BERT model 
download ref :https://github.com/google-research/bert
預訓練模型還提供 BERT-uncased 和 BERT-cased 格式。 在 BERT-uncased 中，所有的 token 都是小寫的，而在 BERT-cased 中，token 不小寫，直接用於訓練。 好的，我們應該使用哪個預訓練的 BERT 模型？ BERT-cased 還是 BERT-uncased？ BERT-uncased 模型是最常用的模型，但是如果我們正在處理某些任務，例如必須保留大小寫的命名實體識別 (NER)，那麼我們應該使用 BERT-cased 模型。 與此同時，谷歌還發布了使用全詞掩碼方法訓練的預訓練 BERT 模型。 好的，但是我們究竟如何使用預訓練的 BERT 模型呢？
we can use the pre-trained model in the following two ways
- as a feature extractor by extracting embeddings 
- by fine-tunning the pre-trained BERT model on downstream task such as text classification.question-answering and more 


## Extracting embeddings from pre-trained BERT 
下載Hugging Face transformers

`pip install transformers==3.5.1`

### generating BERT embeddings 
> chapter3/BERT_embeddings.py
## Extracting embeddings from all encoder layers of  BERT 
> chapter3/BERT_embeddings_allencoder.py
## Fine-tuning BERT for downstream tasks 
how to fine-tune the pre-trained BERT model for the following downstream tasks 
- Text classification
- Natural language inference 
- NER 
- Question-answering 
  
> BERT_sentiment analysis情感分析.py