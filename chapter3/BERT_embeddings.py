from transformers import BertModel,BertTokenizer
import torch

# download the pre-trained model 
model = BertModel.from_pretrained("bert-base-uncased")
# download and load the tokenizer 
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# preprocessing the input 
sentence = 'I love Berlin'
tokens = tokenizer.tokenize(sentence)
print(tokens)
# add ['cls'] and ['sep']
tokens = ['[CLS]'] + tokens + ['[SEP]']

print(tokens)
tokens = tokens+['[PAD]']+['[PAD]']
print(tokens)
# create the attention mask 
attention_mask = [1 if i!=['PAD'] else 0 for i in tokens]
print(attention_mask)
# convert all the tokens to their token IDs 
token_ids = tokenizer.convert_tokens_to_ids(tokens)
print(token_ids)
token_ids = torch.tensor(token_ids).unsqueeze(0)
attention_mask = torch.tensor(attention_mask).unsqueeze(0)
# get the embedding
hidden_rep,cls_head = model(
    token_ids,attention_mask = attention_mask
)
print(hidden_rep)
print(type(hidden_rep))
print(cls_head)
print(type(cls_head))
# print(hidden_rep.shape) # [batch_size,sequence_length,hidden_size]
# print(cls_head.shape)# [batch_size,hidden_size]

