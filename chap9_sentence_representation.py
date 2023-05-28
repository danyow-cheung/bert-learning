from bert_serving.client import BertClient
bc = BertClient()

sentence1 = 'the weather is great today'
sentence2 = 'it looks like today the weather is pretty nice'

sent_rep1 = bc.encode([sentence1])
sent_rep2 = bc.encode([sentence2])

print(sent_rep1.shape, sent_rep2.shape)
from sklearn.metrics.pairwise import cosine_similarity
cosine_similarity(sent_rep1,sent_rep2)
