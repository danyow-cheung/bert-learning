from transformers import BertModel,BertTokenizer
import torch
from transformers import BertConfig, BertModel

# Initializing a BERT bert-base-uncased style configuration
# configuration = BertConfig()

# Initializing a model (with random weights) from the bert-base-uncased style configuration
# model = BertModel(configuration)
# download the pre-trained model 
model = BertModel.from_pretrained("bert-base-uncased",output_hidden_states=True)

# download and load the tokenizer 
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# preprocessing the input 
sentence = 'I love Berlin'
tokens = tokenizer.tokenize(sentence)
print(tokens)
tokens = ['[CLS]'] + tokens + ['[SEP]']
tokens = tokens+['[PAD]']+['[PAD]']

attention_mask = [1 if i!= ['PAD'] else 0 for i in tokens]
token_ids = tokenizer.convert_tokens_to_ids(tokens)
token_ids = torch.tensor(token_ids).unsqueeze(0)
attention_mask = torch.tensor(attention_mask).unsqueeze(0)

res = model(token_ids,attention_mask=attention_mask)


# last_hidden_state,pooler_output,hidden_states = model(token_ids,attention_mask=attention_mask)
last_hidden_state = res[0]
# print(last_hidden_state,type(last_hidden_state))
print(last_hidden_state.shape)
pooler_output = res[1]
print(pooler_output.shape)
hidden_states = res[2]
print(len(hidden_states))
