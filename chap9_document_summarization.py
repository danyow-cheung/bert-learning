from ktrain import text 
import wikipedia 
wiki = wikipedia.page('Pablo Picasso')
docs = wiki.content

print(docs[:100])
ts = text.TransformerSummarizer()

print(ts.summarize(docs))
