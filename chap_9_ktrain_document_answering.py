from ktrain import text
import os 
import shutil
# the first step is initializing the index directory 
text.SimpleQA.initialize_index('index')
text.SimpleQA.index_from_folder(folder_path='bbc',index_dir='index')

qa = text.SimpleQA('index')
answers = qa.ask('who has a global hit with where is the love?')
qa.display_answers(answers[:5])

answers = qa.ask('who win at mtv europe awards?')
qa.display_answers(answers[:5])
