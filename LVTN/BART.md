https://medium.com/mlearning-ai/paper-summary-bart-denoising-sequence-to-sequence-pre-training-for-natural-language-generation-69e41dfbb7fe
https://github.com/jc-hou/train_bart_from_scratch/blob/main/modeling_bart.py
https://github.com/shmsw25/bart-closed-book-qa/blob/master/bart.py
https://www.geeksforgeeks.org/bart-model-for-text-auto-completion-in-nlp/
***BART Overview***

BART is a denoising autoencoder that maps a corrupted
document to the original document it was derived from.
It is implemented as a sequence-to-sequence model
with a bidirectional encoder over corrupted text and a
left-to-right autoregressive decoder. For pre-training,
we optimize the negative log likelihood of the original
document.
The [original Transformer](https://arxiv.org/abs/1706.03762 "original Transformer") is based on an encoder-decoder architecture and is a classic sequence-to-sequence model. The model’s input and output are in the form of a sequence (text), and the encoder learns a high-dimensional representation of the input,which is then mapped to the output by the decoder. This architecture introduced a new form of learning for language-related tasks and, thus, the models spawned from it achieve outstanding results overtaking the existing deep [neural network-based methods](https://www.projectpro.io/article/neural-network-projects/440 "neural network-based methods").

Since the inception of the vanilla Transformer, several recent models inspired by the Transformer used the architecture to improve the benchmark of [NLP tasks](https://www.projectpro.io/article/10-nlp-techniques-every-data-scientist-should-know/415 "NLP tasks"). Transformer models are first pre-trained on a large text corpus (such as BookCorpus or Wikipedia). This pretraining makes sure that the model “understands language” and has a decent starting point to learn how to perform further tasks. Hence, after this step, we only have a language model. The ability of the model to understand language is highly significant since it will determine how well you can further train the model for something like text classification or text summarization.

BART model is one such Transformer model that takes components from other Transformer models and improves the pretraining learning. BART or Bidirectional and Auto-Regressive Transformers was proposed in the [BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension](https://arxiv.org/abs/1910.13461 "BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension") paper. The [BART HugggingFace model](https://huggingface.co/docs/transformers/model_doc/bart "BART HugggingFace model") allows the pre-trained weights and weights fine-tuned on question-answering, text summarization, conditional text generation, mask filling, and sequence classification.
***Architecture***
BART uses the standard sequence-to-sequence Trans-
former architecture from (Vaswani et al., 2017), ex-
cept, following GPT, that we modify ReLU activa-
tion functions to GeLUs (Hendrycks & Gimpel, 2016)
and initialise parameters from N (0, 0.02). For our
base model, we use 6 layers in the encoder and decoder, and for our large model we use 12 layers in
each. The architecture is closely related to that used in
BERT, with the following differences: (1) each layer of
the decoder additionally performs cross-attention over
the final hidden layer of the encoder (as in the trans-
former sequence-to-sequence model); and (2) BERT
uses an additional feed-forward network before word-
prediction, which BART does not. In total, BART con-
tains roughly 10% more parameters than the equiva-
lently sized BERT model.


To understand the purpose and philosophy behind BART architecture, first, we need to understand the nature of the NLP tasks it set out to solve. In tasks, such as question answering and text summarization, that require natural language understanding (or NLU) it is imperative for our model to read the text as a whole and understand each token in the context of what came, both, before and after it. For instance, training a masked language model with the sentence “the man went to the dairy store to buy a gallon of milk” could have a sentence like this as input based on the nosing schemes we saw above: 

“the man went to the [MASK] store to buy a gallon of milk”.

Now, for an NLU task, it is important for the model to read the sentence completely before predicting [MASK] since it highly depends on the terms like “store” and “milk”. In such a case, the input sequence can be properly interpreted and learned by a bi-directional approach to reading and representing the text. The BERT (or Bidirectional Encoder Representations from Transformers) model incorporates this idea to greatly improve the language modeling task that happens in pre-training.

Thus, the first part of BART architecture uses the bi-directional encoder of BERT to find the best representation of its input sequence. For every text sequence in its input, the BERT encoder outputs an embedding vector for each token in the sequence as well as an additional vector containing sentence-level information. In this way, the decoder can learn for both token and sentence-level tasks making it a robust starting point for any future fine-tuning tasks. 

The pre-training is done using the masked sequences as discussed previously and shown below. While [BERT](https://www.projectpro.io/project-use-case/multi-class-text-classification-using-bert "BERT") was trained by using a simple token masking technique, BART empowers the BERT encoder by using more challenging kinds of masking mechanisms in its pre-training.

![](https://dezyre.gumlet.io/images/blog/transformers-bart-model-explained/image_190005529141642833004007.png?w=940&dpr=1.3)

Once we get the token and sentence-level representation of an input text sequence, a decoder needs to interpret these to map with the output target. However, by using a similarly designed decoder, tasks such as next sentence prediction or token prediction might perform poorly since the model relies on a more comprehensive input prompt. In these cases, we need model architectures that can be trained on generating the next word by only looking at the previous words in the sequence. Hence, a causal or autoregressive model that looks only at the past data to predict the future comes in handy. 

![](https://dezyre.gumlet.io/images/blog/transformers-bart-model-explained/image_645899080121642833004001.png?w=940&dpr=1.3)

The GPT-1 model used an architecture similar to the decoder segment of the vanilla Transformers. GPT sequentially stacks 12 such decoders such that learning from only the past tokens can affect the current token calculation. The architecture is shown above. As seen in the original Transformer decoder, the GPT decoder also uses a masked multiheaded self-attention block and a feed-forward layer.

**Get FREE Access to [Machine Learning Example Codes](https://www.projectpro.io/recipes?utm_source=TXTCTA2&utm_medium=RcpLink&utm_campaign=blg553 "Data Science and Machine Learning Python Example Codes") for Data Cleaning, Data Munging, and Data Visualization**

Comparable to other models we discussed here, including BART, GPT also takes a semi-supervised approach to learning. First, the model is pre-trained on tokens “t” looking back to “k” tokens in the past to compute the current token. This is done unsupervised on a vast text corpus to allow the model to “learn the language.”

![](https://dezyre.gumlet.io/images/blog/transformers-bart-model-explained/image_129641438171642833004043.png?w=940&dpr=1.3)

Next, to make the model robust on a specific task, it is fine-tuned in a supervised manner to maximize the likelihood of label “y” given feature vectors x1…xn.

![](https://dezyre.gumlet.io/images/blog/transformers-bart-model-explained/image_322903976191642833004171.png?w=940&dpr=1.3)

Combining 1 and 2, we get the objective in 3. Lambda represents a learned weight parameter to control the influence of language modeling.

![](https://dezyre.gumlet.io/images/blog/transformers-bart-model-explained/blobid0.png?w=940&dpr=1.3)

Below image shows how the autoregressive decoder processes its input.

![](https://dezyre.gumlet.io/images/blog/transformers-bart-model-explained/image_649541961151642833004027.png?w=940&dpr=1.3)

Although we separate the decoder from an encoder, the input to the decoder would still be a learned representation (or embedding) of the original text sequence. Thus, BART attaches the bi-directional encoder to the autoregressive decoder to create a denoising auto-encoder architecture. And based on these two components, the final BART model would look something like this:

![](https://dezyre.gumlet.io/images/blog/transformers-bart-model-explained/image_10036092571642833003977.png?w=940&dpr=1.3)

In the above figure, the input sequence is a masked (or noisy) version of [ABCDE] transformed into [A[MASK]B[MASK]E]. The encoder looks at the entire sequence and learns high-dimensional representations with bi-directional information. The decoder takes these thought vectors and regressively predicts the next token. Learning occurs by computing and optimizing the negative log-likelihood as mapped with the target [ABCDE].








**BART** stands for Bidirectional and Auto-Regressive Transformer. It is a denoising autoencoder that is a pre-trained sequence-to-sequence method, that uses masked language modeling for Natural Language Generation and Translation. It is developed by Lewis et al. in 2019. BART architecture is similar to an encoder-decoder network except that it uses a combination of BERT and GPT models. The BART models can be fine-tuned over small supervised datasets to create domain-specific tasks. 

### Denoising autoencoder

An autoencoder is a special type of neural network that learns to encode an input sentence into lower dimensional representations and decode the embedded representations back to the corresponding original input sentences. In a general case, when the input and output sentence of an autoencoder is the same, over a large number of iterations, the autoencoder network directly maps the input token to the output tokens, and the embedded representation that is usually learned between them becomes redundant. Therefore, we modify the input sentence by randomly deleting word tokens and replacing them with a special **<MASK>** token, this sentence with the randomly deleted token is called a **corrupted or noisy sentence** and the **supervised** output for the corresponding input is the clean sentence with all the original tokens preserved. By learning to predict the missing or corrupted tokens, the denoising autoencoder learns to extract meaningful features from the input sentence. **A denoising autoencoder is trained on a large corpus of such data so it learns to predict the masked/deleted token in the input sentence which is responsible for the noise in the text, as a result, we get a clean and semantically coherent output, hence the term “denoising” is added to the autoencoder.**

## BART (Bidirectional and Auto-Regressive Transformer) Architecture

For a given input text sequence, the [BERT (Bidirectional Representation for Transformers)](https://www.geeksforgeeks.org/understanding-bert-nlp/) encoder network generates an [embedding](https://www.geeksforgeeks.org/word-embeddings-in-nlp/) for each token in the input text and an additional sentence-level embedding vector. The **GPT decoder** network learns this token-level and sentence-level embedded information and its existing pre-trained weights to generate clean semantically close text sequences.

**BART** has approximately 140 million parameters which are greater than **BERT** (110 million parameters) and **GPT-1** (117 million) models but outperform them significantly given that **BART** is a combination of them both. 

**BART’s** primary task is used to generate clean semantically coherent text from corrupted text data but it can also be used for a variety of different NLP sub-tasks like language translation, question-answering tasks, text summarization, paraphrasing, etc.

As BART is an autoencoder model, it consists of an **encoder model** and a **decoder model**. For its encoder model, BART uses a  **bi-directional encoder** that is used in [BERT](https://www.geeksforgeeks.org/understanding-bert-nlp/), and for its decoder mode, it uses an **autoregressive decoder** that forms the core aspect of a [GPT](https://www.geeksforgeeks.org/open-ai-gpt-3/) -1 model. 

An **autoregressive decoder** is a neural network architecture that takes the previous input tokens as well as the current token to predict the next token at every time step. It is important to remember that the input accepted by a decoder is an embedding created by its corresponding encoder network.

Both the encoder and decoder architecture is built by the combination of multiple blocks or layers where each block processes information in a specific way.

It consists of 3 primary blocks:

- Multi-head Attention block
- Addition and Normalization block
- Feed-forward layers

#### Multi-head attention block

This is one of the most important blocks as in this layer multiple levels of masking( replacing random tokens in a sentence with the <MASK> token ) are performed over the predicting tokens, for example:

**Parallel #1 thread**: Entire sentence is replaced by the <MASK> tokens.

**Parallel #2 thread**: Multiple bi-gram tokens are replaced by the <MASK> tokens.

**Parallel #3+ thread**: Arbitrary words within the sentence are replaced by the <MASK> token. 

This masking is done in parallel instead of sequentially to avoid accumulating previous step errors for the same input sentence.

#### Addition and Normalization block

Different parameters within the multiple blocks contain values within different ranges, hence to add those values together, we scale the values of all the parameters into a single range using a monotonic function whose value converges to a constant value _k_ as the input closes to infinity. This is performed so that uniform weight for all parameters is ensured while concatenating multiple parameters into a single one.

#### Feed-forward Layers

The [feed-forward](https://www.geeksforgeeks.org/introduction-to-ann-set-4-network-architectures/) layers compose the basic building block of any neural network and are composed of hidden layers containing a fixed number of [neurons](https://www.geeksforgeeks.org/single-neuron-neural-network-python/). These layers contain the process, and store information coming from the previous layers as weights and forward the processed/ updated information to the next layer. The [feed-forward neural network layers](https://www.geeksforgeeks.org/understanding-multi-layer-feed-forward-networks/) are specially designed to move information in a sequential uni-directional manner.

![BART -Geeksforgeeks](https://media.geeksforgeeks.org/wp-content/uploads/20230321032520/bart1drawio-(2).png)

BART single encoder-decoder network architecture

### BERT Encoder Cell

[BERT](https://www.geeksforgeeks.org/explanation-of-bert-model-nlp/) each encoder cell contains a multi-head attention that accepts raw tokenized text, any other preprocessing (lowercase, stemming, stopwords removal, etc) of the text is completely task-dependent. The tokenized text is then uni-label encoded and passed into the multi-head attention block where the text tokens are randomly masked and forwarded to the add and norm layer. We use a skip connection from the input layer to combine both the complete clean text as well as randomly masked tokens and after multiple such iterations, we then pass the information into the standard feed-forward block which also adds and normalizes the current as well as the original information from the first “add and normalization” layer. 

The result is an embedding containing clean, masked, and compressed information regarding the original input text. 

The [skip connections](https://www.geeksforgeeks.org/residual-networks-resnet-deep-learning/) are important to remember clean unprocessed information as well as newly processed information when the data is passed through each block. 

Another important addition to processing text using a [BERT](https://www.geeksforgeeks.org/explanation-of-bert-model-nlp/) encoder cell is that it is bi-directional i.e. the tokens are passed through the encoder cell from the beginning of the sentence to the end and vice versa (both the direction of the text). This is done so that information learned at the end of the sentence should also be able to adjust the weight of the embedding and cause a semantic change at the beginning of the output tokens if required.

###  GPT Decoder Cell

The GPT decoder cell accepts masked embeddings from the [**BERT**](https://www.geeksforgeeks.org/understanding-bert-nlp/) **Encoder cell** and passes it to the **masked multiple self-attention block** which follows the same architecture as the **multi-head attention block** but works sequentially instead of parallel, where instead of learning the encoding of the masks, this layer learns to decode the masked embeddings to semantically coherent tokens by paying attention to the multiple different parallel embeddings of the input text masked at different levels. 

Following this, the output is added and normalized with the original embedding and passed to the standard **feed-forward layer** block where information is learned to decode the sentence from the processed embedding, and at the final layer, we try to match the predicted tokens with the clean output which is the backtracked over the entire encoder-decoder network. This training process continues over a very large corpus of examples, that not only learn the context of the sentences but also greatly improve the network’s capability in predicting missing <MASK> tokens, i.e. helps the network clean real-life corrupted sentences.