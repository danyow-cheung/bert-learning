代码仓库：
https://github.com/PacktPublishing/Getting-Started-with-Google-BERT

## Chapter 1 ,a primer on transfroms

​	transfroms 取代rnn和lstm在nlp界

### Introduction to the transformer 

​	transformer的出现在NLP领域创造了重大突破，也为新的革命性架构（如BERT、GPT-3、T5等）铺平了道路



​	transformer基于<u>注意力机制，完全避免了重复</u>。transforer使用了一种特殊的注意力机制，称为自我注意力



### Understanding the encoder of the transformer 

​	变压器由一堆编码器组成。一个编码器的输出作为输入发送到其上方的编码器。

![截圖 2022-09-12 16.53.20](/Users/danyow/Library/Application Support/typora-user-images/截圖 2022-09-12 16.53.20.png)

​	假设N=2,可以得到以下的编码器

![截圖 2022-09-12 16.56.29](/Users/danyow/Library/Application Support/typora-user-images/截圖 2022-09-12 16.56.29.png)

​	编码器有两个组成

- multi-head attention多头注意力
- feedforward network前馈网络

#### self-attention mechanism 自注意力机制

​	给出句子

> a dog ate the food because it was hungry

在这句话中，it可以指代狗或者食物，通过阅读句子，我们可以知道it指代狗。但是模型怎么知道？

​	首先，我们的模型计算单词A的表示，然后计算单词dog的表示，再计算单词ate的表示，依此类推。在计算每个单词的表示时，它将每个单词与句子中的所有其他单词联系起来，以了解更多关于单词的信息。也就是说，模型会把it和所有单词联系起来，以更了解it

​	

​	**实现方法**

	1. get the embeddings for each word 
	1. use embedding to represent the input sentence

​	因此，input matrix 的shape[句子长度x嵌入维数],number of words is N,设置嵌入维数为512，所以input matrix 就是[Nx512]



​	<u>从输入矩阵中，我们创建了三个新矩阵：查询矩阵(query matrix)、键矩阵(Q key matrix)和值矩阵(value matrix)</u>

​	创建矩阵，先有3个权重参数，叫做 *Wq,Wk,Wv*。分别乘以参数得到输入矩阵。

​	权重参数随机初始化，并在训练中会进行微调。当我们学习最佳权重时，我们将获得更精确的查询、键和值矩阵。

​	![截圖 2022-09-12 17.29.15](/Users/danyow/Library/Application Support/typora-user-images/截圖 2022-09-12 17.29.15.png)

请注意，查询、键和值向量的维数为64。因此，查询、键值矩阵的维数为[句子长度x 64]。由于句子中有三个词，因此查询、键和值矩阵的维度为[3 x 64]

#### Understanding the self-attention mechanism

我们了解到，为了计算单词的表示，自我注意机制将单词与给定句子中的所有单词联系起来。考虑一下“我很好”这句话。为了计算单词I的表示，我们将单词I与句子中的所有单词关联起来，

![截圖 2022-09-12 17.39.49](/Users/danyow/Library/Application Support/typora-user-images/截圖 2022-09-12 17.39.49.png)

但我们为什么要这样做？理解一个单词与句子中所有单词的关系有助于我们学习更好的表达。现在，让我们学习自我注意机制如何使用查询、键和值矩阵将单词与句子中的所有单词联系起来。自我注意机制包括四个步骤；让我们一个一个地看一看。

1. the first step in the self-attention mechanism is to compute the dot prodcut between the query matrix ,Q and the key matrix,K^T

   > 但是計算矩陣到底有什麼用，K^T到底是啥？
   >
   > 計算query和key兩個矩陣是為了知道兩個向量的相似性
   >
   > 

2. the second step is to divide the <u>Q·K^T</u> matrix by the square root of the dimension of the key vector 將<u>Q·K^T</u>矩陣除以關鍵向量維數的平方根。為的是獲得更加穩定的梯度。

3. by looking at the preceding similarity scores,the matrix is umnormalized form,so we are about to normalize them using the <u>softmax</u> function.

4. the final step in the self-attention mechanism is to compute the attention matrix,Z

   > the attention matrix contains the attention values for the each word in the sentence.
   >
   > 
   >
   > The attention matrix, Z, is computed by taking the sum of the value vectors weighted by the scores. 

   ![截圖 2022-10-07 16.42.54](/Users/danyow/Library/Application Support/typora-user-images/截圖 2022-10-07 16.42.54.png)	 	

   

   

   

   

   #### multi-head attention mechanism

   這僅在實際單詞的含義為模糊的。 也就是說，考慮以下句子：

   *a dog ate the food because it was hungry*

   可以增加句子預測的準確性

   #### learning position with positional encoding

   在傳統的rnn中，句子輸入為<u>iamgood</u>時，會依次輸入單詞以至於模型可以理解句子。

   但是在transoform網絡中，我們不遵循回溯機制。我們不是逐字輸入句子，而是將句子中的所有單詞並行輸入到網絡中。 並行輸入單詞有助於減少訓練時間，也有助於學習長期依賴。

   

   **但是在理解句子過程中，句子的語言順序很重要。**

   

### Understanding the decoder of the transformer 

英文轉換為法文，解碼器是用來輸出法文的



we can observe that the decoder block is similar to the encoder and here we have three sublayers:

- Masked multi-head attention 面具的多頭注意力
- Multi-head attention 多頭注意力
- Feedforward network  前饋網路







## Chapter 2, Understanding the BERT Model

the BERT model is pre-trained using two tasks, called masked language modeling and next sentence prediction,

### Basic idea of BERT 

BERT stands for Bidirectional Encoder Repression from Transformer 來自 Transformer 的雙向編碼器抑制

而且BERT成功的關鍵是 context-based embedding model 

### Configurations of BERT 

two standard configurations 

- BERT-base

  BERT-base consists of **12** encoder layers, each stacked one on top of the other. All the encoders use **12** attention heads. The feedforward network in the encoder consists of **768** hidden units. Thus, the size of the representation obtained from BERT-base will be 768.

  BERT-base 由 **12** 編碼器層組成，每個編碼器層堆疊在一起。 所有編碼器都使用 **12** attention heads。 編碼器中的前饋網絡由 **768** 隱藏單元組成。 因此，從 BERT-base 獲得的表示的大小將為 768。

  

- BERT-large

  BERT-large consists of **24** encoder layers, each stacked one on top of the other. All the encoders use **16** attention heads. The feedforward network in the encoder consists
   of **1,024** hidden units. Thus, the size of the representation obtained from BERT-large will be 1,024.

  BERT-large 由 24 個編碼器層組成，每個編碼器層堆疊在一起。 所有編碼器都使用 16 個注意力頭。 編碼器中的前饋網絡包括
  1,024 個隱藏單元。 因此，從 BERT-large 獲得的表示的大小將為 1,024。

### pre-training the BERT model 

#### Input data representation 

Before feeding the input to BERT, we convert the input into embeddings using the three embedding layers indicated here:

- Token embedding
- Segment embedding 

- Position embedding



##### Tokenembedding

```python
tokens = [Paris,is,a,beautiful,city]
tokens = [[CLS],Paris,is,a,beautiful,city]
tokens = [[CLS]Paris,is,a,beautiful,city,[SEP]]
```

##### Segment embedding

The segment embedding layer returns only either of the two embeddings, `E_{b}` or `E_{a}`, as an output. 



##### WordPiece tokenizer

When we tokenize using the WordPiece tokenizer, first we check whether the word is present in our vocabulary. If the word is present in the vocabulary, then we use it as a token. If the word is not present in the vocabulary, then we split the word into
 subwords and we check whether the subword is present in the vocabulary. If the subword is present in the vocabulary, then we use it as a token. But if the subword is not present in the vocabulary, then again we split the subword and check whether it is present in the vocabulary. If it is present in the vocabulary, then we use it as a token, otherwise we split it again. In this way, we keep splitting and check the subword with the vocabulary until we reach individual characters. This is effective in handling the **out-of-vocabulary**

(**OOV**) words.

當我們使用 WordPiece 分詞器進行分詞時，首先我們檢查該詞是否存在於我們的詞彙表中。 如果該詞出現在詞彙表中，那麼我們將其用作標記。 如果單詞不存在於詞彙表中，那麼我們將單詞拆分為
子詞，我們檢查子詞是否存在於詞彙表中。 如果子詞存在於詞彙表中，那麼我們將其用作標記。 但是如果子詞不存在於詞彙表中，那麼我們再次拆分子詞並檢查它是否存在於詞彙表中。 如果它存在於詞彙表中，那麼我們將它用作標記，否則我們再次拆分它。 以這種方式，我們不斷拆分並用詞彙表檢查子詞，直到我們到達單個字符。 這在處理詞彙外的問題時很有效
（OOV）的話。













## Chapter 3, Getting Hands*–*On with BERT

> matter 
>
> 

