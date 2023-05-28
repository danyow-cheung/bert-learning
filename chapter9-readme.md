# Working with VideoBERT,BART and More 


## Performing text summarization with BART 
```python
from transformers import BartTokenizer, BartForConditionalGeneration
model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')


text = """Machine learning (ML) is the study of computer algorithms that
improve automatically through experience.It is seen as a subset of
artificial intelligence. Machine learning algorithms build a mathematical
model based on sample data, known as training data, in order to make
predictions or decisions without being explicitly programmed to do
so.Machine learning algorithms are used in a wide variety of applications,
such as email filtering and computer vision, where it is difficult or
infeasible to develop conventional algorithms to perform the needed
tasks."""
inputs = tokenizer([text],max_length=1024,return_tensors='pt')
summary_ids = model.generate(inputs['input_ids'],num_beams=4,max_length=100,early_stopping=True)
summary = ([tokenizer.decode(i, skip_special_tokens=True,clean_up_tokenization_space=False) for i in summary_idx])
print(summary)

```

## Sentiment analysis using ktrain 
> chap_9_ktrain_Sentiment_analysis.py

## Building a document answering model 
> chap_9_ktrain_document_answering.py
>
## Document summarization
> chap9_document_summarization.py
>
## Bert-as-service 
### computing sentence represstation
> chap9_sentence_representation.py
>
### Computing contextual word represstation
> chap9_contextual_representation.py
>