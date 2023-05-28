import ktrain
from ktrain import text
import pandas as pd 

df = pd.read_json(r'reviews_Digital_Music_5.json',lines=True)
# print(df.head())

df = df[['reviewText','overall']]

sentiment = {1:'negative',2:'negative',3:"negative",4:'postive',5:'postive'}

df['sentiment'] = df['overall'].map(sentiment)
df = df[['reviewText','sentiment']]
print('8'*20)
print(df)
print('8'*20)

(x_train,y_train),(x_test,y_test) ,preproc = ktrain.text.texts_from_df(
    train_df=df,text_column='reviewText',
    label_columns=['sentiment'],
    maxlen=100,
    max_features=100000,
    preprocess_mode='bert',
    val_pct = 0.1 
    )
text.print_text_classifiers()

model = text.text_classifier(name='bert', train_data = (x_train,y_train),preproc=preproc,metrics=['accuracy'])
learner = ktrain.get_learner(
    model=model,
    train_data=(x_train,y_train),
    val_data=(x_test,y_test),
    batch_size=32,
    use_multiprocessing=True
)
learner.fit_onecycle(lr=2e-5,epochs=1,checkpoint_folder='output')

predictor = ktrain.get_predictor(learner.model, preproc)
print(predictor.predict('i love the song'))
