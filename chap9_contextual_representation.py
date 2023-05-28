from bert_serving.client import BertClient
bc = BertClient()
sentence = 'The weather is great today'
vec = bc.encode([sentence])
print(vec.shape)
