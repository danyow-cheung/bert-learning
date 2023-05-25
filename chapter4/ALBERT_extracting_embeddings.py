from transformers import AlbertModel,AlbertTokenizer
model = AlbertModel.from_pretrained('albert-base-v2')
tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
sentence = "Paris is a beautiful city"
inputs = tokenizer(sentence,return_tensors='pt')
print(inputs)

hidden_rep, cls_head = model(**inputs)
print(hidden_rep)

