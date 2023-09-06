tokenizing a text is splitting it into words or subwords, which then are converted to ids through a look-up table. Converting words or subwords to ids is straightforward, so in this summary, we will focus on splitting a text into words or subwords (i.e. tokenizing a text)

Byte-Pair Encoding (BPE) was introduced in [Neural Machine Translation of Rare Words with Subword Units (Sennrich et al., 2015)](https://arxiv.org/abs/1508.07909). BPE relies on a pre-tokenizer that splits the training data into words. Pretokenization can be as simple as space tokenization, e.g. [GPT-2](https://huggingface.co/docs/transformers/model_doc/gpt2), [RoBERTa](https://huggingface.co/docs/transformers/model_doc/roberta). More advanced pre-tokenization include rule-based tokenization, e.g. [XLM](https://huggingface.co/docs/transformers/model_doc/xlm), [FlauBERT](https://huggingface.co/docs/transformers/model_doc/flaubert) which uses Moses for most languages, or [GPT](https://huggingface.co/docs/transformers/model_doc/gpt) which uses Spacy and ftfy, to count the frequency of each word in the training corpus.

After pre-tokenization, a set of unique words has been created and the frequency with which each word occurred in the training data has been determined. Next, BPE creates a base vocabulary consisting of all symbols that occur in the set of unique words and learns merge rules to form a new symbol from two symbols of the base vocabulary. It does so until the vocabulary has attained the desired vocabulary size. Note that the desired vocabulary size is a hyperparameter to define before training the tokenizer.

As an example, let’s assume that after pre-tokenization, the following set of words including their frequency has been determined:

("hug", 10), ("pug", 5), ("pun", 12), ("bun", 4), ("hugs", 5)

Consequently, the base vocabulary is `["b", "g", "h", "n", "p", "s", "u"]`. Splitting all words into symbols of the base vocabulary, we obtain:

("h" "u" "g", 10), ("p" "u" "g", 5), ("p" "u" "n", 12), ("b" "u" "n", 4), ("h" "u" "g" "s", 5)

BPE then counts the frequency of each possible symbol pair and picks the symbol pair that occurs most frequently. In the example above `"h"` followed by `"u"` is present _10 + 5 = 15_ times (10 times in the 10 occurrences of `"hug"`, 5 times in the 5 occurrences of `"hugs"`). However, the most frequent symbol pair is `"u"` followed by `"g"`, occurring _10 + 5 + 5 = 20_ times in total. Thus, the first merge rule the tokenizer learns is to group all `"u"` symbols followed by a `"g"` symbol together. Next, `"ug"` is added to the vocabulary. The set of words then becomes

("h" "ug", 10), ("p" "ug", 5), ("p" "u" "n", 12), ("b" "u" "n", 4), ("h" "ug" "s", 5)

BPE then identifies the next most common symbol pair. It’s `"u"` followed by `"n"`, which occurs 16 times. `"u"`, `"n"` is merged to `"un"` and added to the vocabulary. The next most frequent symbol pair is `"h"` followed by `"ug"`, occurring 15 times. Again the pair is merged and `"hug"` can be added to the vocabulary.

At this stage, the vocabulary is `["b", "g", "h", "n", "p", "s", "u", "ug", "un", "hug"]` and our set of unique words is represented as

("hug", 10), ("p" "ug", 5), ("p" "un", 12), ("b" "un", 4), ("hug" "s", 5)

Assuming, that the Byte-Pair Encoding training would stop at this point, the learned merge rules would then be applied to new words (as long as those new words do not include symbols that were not in the base vocabulary). For instance, the word `"bug"` would be tokenized to `["b", "ug"]` but `"mug"` would be tokenized as `["<unk>", "ug"]` since the symbol `"m"` is not in the base vocabulary. In general, single letters such as `"m"` are not replaced by the `"<unk>"` symbol because the training data usually includes at least one occurrence of each letter, but it is likely to happen for very special characters like emojis.

As mentioned earlier, the vocabulary size, _i.e._ the base vocabulary size + the number of merges, is a hyperparameter to choose. For instance [GPT](https://huggingface.co/docs/transformers/model_doc/gpt) has a vocabulary size of 40,478 since they have 478 base characters and chose to stop training after 40,000 merges.

#### [](https://huggingface.co/docs/transformers/tokenizer_summary#bytelevel-bpe)Byte-level BPE

A base vocabulary that includes all possible base characters can be quite large if _e.g._ all unicode characters are considered as base characters. To have a better base vocabulary, [GPT-2](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) uses bytes as the base vocabulary, which is a clever trick to force the base vocabulary to be of size 256 while ensuring that every base character is included in the vocabulary. With some additional rules to deal with punctuation, the GPT2’s tokenizer can tokenize every text without the need for the symbol. [GPT-2](https://huggingface.co/docs/transformers/model_doc/gpt) has a vocabulary size of 50,257, which corresponds to the 256 bytes base tokens, a special end-of-text token and the symbols learned with 50,000 merges.

# Subword-based tokenization

Subword-based tokenization is a solution between word and character-based tokenization. 😎 The main idea is to solve the issues faced by word-based tokenization (very large vocabulary size, large number of OOV tokens, and different meaning of very similar words) and character-based tokenization (very long sequences and less meaningful individual tokens).

The subword-based tokenization algorithms do not split the frequently used words into smaller subwords. It rather splits the rare words into smaller meaningful subwords. For example, “boy” is not split but “boys” is split into “boy” and “s”. This helps the model learn that the word “boys” is formed using the word “boy” with slightly different meanings but the same root word.

Some of the popular subword tokenization algorithms are WordPiece, Byte-Pair Encoding (BPE), Unigram, and SentencePiece. We will go through Byte-Pair Encoding (BPE) in this article. BPE is used in language models like GPT-2, RoBERTa, XLM, FlauBERT, etc. A few of these models use space tokenization as the pre-tokenization method while a few use more advanced pre-tokenization methods provided by Moses, spaCY, ftfy. So, let’s get started. 🏃

# Byte-Pair Encoding (BPE)

BPE is a simple form of data compression algorithm in which the most common pair of consecutive bytes of data is replaced with a byte that does not occur in that data. It was first described in the article “[A New Algorithm for Data Compression](https://www.drdobbs.com/a-new-algorithm-for-data-compression/184402829)” published in 1994. The below example will explain BPE and has been taken from [Wikipedia](https://en.wikipedia.org/wiki/Byte_pair_encoding).

Suppose we have data **aaabdaaabac** which needs to be encoded (compressed). The byte pair **aa** occurs most often, so we will replace it with **Z** as **Z** does not occur in our data. So we now have **ZabdZabac** where **Z = aa**. The next common byte pair is **ab** so let’s replace it with **Y**. We now have **ZYdZYac** where **Z = aa** and **Y = ab**. The only byte pair left is **ac** which appears as just one so we will not encode it. We can use recursive byte pair encoding to encode **ZY** as **X**. Our data has now transformed into **XdXac** where **X = ZY, Y = ab,** and **Z = aa**. It cannot be further compressed as there are no byte pairs appearing more than once. We decompress the data by performing replacements in reverse order.

A variant of this is used in NLP. Let us understand the NLP version of it together. 🤗

==BPE ensures that the most common words are represented in the vocabulary as a single token while the rare words are broken down into two or more subword tokens and this is in agreement with what a subword-based tokenization algorithm does.==

Suppose we have a corpus that has the words (after pre-tokenization based on space) — old, older, highest, and lowest and we count the frequency of occurrence of these words in the corpus. Suppose the frequency of these words is as follows:

**{“old”: 7, “older”: 3, “finest”: 9, “lowest”: 4}**

Let us add a special end token “</w>” at the end of each word.

**{“old</w>”: 7, “older</w>”: 3, “finest</w>”: 9, “lowest</w>”: 4}**

The “</w>” token at the end of each word is added to identify a word boundary so that the algorithm knows where each word ends. This helps the algorithm to look through each character and find the highest frequency character pairing. I will explain this part in detail later when we will include “</w>” in our byte-pairs.

Moving on next, we will split each word into characters and count their occurrence. The initial tokens will be all the characters and the “</w>” token.

![](https://miro.medium.com/v2/resize:fit:875/1*5MEIKtS02pU9mO7M_Mp_cQ.png)

Since we have 23 words in total, so we have 23 “</w>” tokens. The second highest frequency token is “e”. In total, we have 12 different tokens.

The next step in the BPE algorithm is to look for the most frequent pairing, merge them, and perform the same iteration again and again until we reach our token limit or iteration limit.

Merging lets you represent the corpus with the least number of tokens which is the main goal of the BPE algorithm, that is, compression of data. To merge, BPE looks for the most frequently represented byte pairs. Here, we are considering a character to be the same as a byte. This is a case in the English language and can vary in other languages. Now we will merge the most common bye pairs to make one token and add them to the list of tokens and recalculate the frequency of occurrence of each token. This means our frequency count will change after each merging step. We will keep on doing this merging step until we hit the number of iterations or reach the token limit size.

## Encoding and Decoding

Let us now see how we will decode our example. To decode, we have to simply concatenate all the tokens together to get the whole word. For example, the encoded sequence [“the</w>”, “high”, “est</w>”, “range</w>”, “in</w>”, “Seattle</w>”], we will be decoded as [“the”, “highest”, “range”, “in”, “Seattle”] and not as [“the”, “high”, “estrange”, “in”, “Seattle”]. Notice the presence of the “</w>” token in “est”.

For encoding the new data, the process is again simple. However, encoding in itself is computationally expensive. Suppose the sequence of words is [“the</w>”, “highest</w>”, “range</w>”, “in</w>”, “Seattle</w>”]. We will iterate through all the tokens we found in our corpus — longest to the shortest and try to replace substrings in our given sequence of words using these tokens. Eventually, we will iterate through all the tokens and our substrings will be replaced with tokens already present in our token list. If a few substrings will be left (for words our model did not see in training), we will replace them with unknown tokens.

In general, the vocabulary size is big but still, there is a possibility of an unknown word. In practice, we save the pre-tokenized words in a dictionary. For unknown (new) words, we apply the above-stated encoding method to tokenize the new word and add the tokenization of the new word to our dictionary for future reference. This helps us build our vocabulary even stronger for the future.