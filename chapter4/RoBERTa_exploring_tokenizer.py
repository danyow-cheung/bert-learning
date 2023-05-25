from transformers import RobertaModel,RobertaConfig,RobertaTokenizer

model = RobertaModel.from_pretrained('roberta-base')

print(model.config)
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
print(tokenizer.tokenize('It was a great day'))
