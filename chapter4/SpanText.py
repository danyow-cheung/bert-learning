'''
Performing Q&As with pre-trained SpanBERT
'''
from transformers import pipeline

qa_pipeline = pipeline(
    'question-answering',
    model="mrm8488/spanbert-large-finetuned-squadv2",
    tokenizer="SpanBERT/spanbert-large-cased"
)
results = qa_pipeline({
    'question': "What is machine learning?",
    'context': "Machine learning is a subset of artificial intelligence. It is widely for creating a variety of applications such as email filtering and computer vision"

})

print(results['answer'])
