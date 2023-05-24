from transformers import BertForQuestionAnswering,BertTokenizer

model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-fine-tuned-squad')

tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-fine-tuned-squad')

