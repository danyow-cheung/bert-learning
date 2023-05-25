from transformers import ElectraTokenizer,ElectraModel

model = ElectraModel.from_pretrained('google/electra-small-discriminator')
