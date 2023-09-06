[**ROUGE** (Recall-Oriented Understudy for Gisting Evaluation)](https://aclanthology.org/W04-1013.pdf), is a set of metrics and a software package specifically designed for evaluating [automatic summarization](https://en.wikipedia.org/wiki/Automatic_summarization), but that can be also used for [machine translation](https://en.wikipedia.org/wiki/Machine_translation). The metrics compare an automatically produced summary or translation against reference (high-quality and human-produced) summaries or translations.

In this article, we cover the main metrics used in the ROUGE package.

# ROUGE-N

**ROUGE-N** measures the number of matching [n-grams](https://en.wikipedia.org/wiki/N-gram) between the model-generated text and a human-produced reference.

Consider the reference _R_ and the candidate summary _C_:

- _R_: The cat is on the mat.
- _C_: The cat and the dog.

## ROUGE-1

Using _R_ and _C_, we are going to compute the [precision, recall, and F1-score](https://blog.exsilio.com/all/accuracy-precision-recall-f1-score-interpretation-of-performance-measures) of the matching n-grams. Let’s start computing ROUGE-1 by considering 1-grams only.

ROUGE-1 precision can be computed as the ratio of the number of unigrams in _C_ that appear also in _R_ (that are the words “the”, “cat”, and “the”), over the number of unigrams in _C_.

> ROUGE-1 precision = 3/5 = 0.6

ROUGE-1 recall can be computed as the ratio of the number of unigrams in _R_ that appear also in _C_ (that are the words “the”, “cat”, and “the”), over the number of unigrams in _R_.

> ROUGE-1 recall = 3/6 = 0.5

Then, ROUGE-1 F1-score can be directly obtained from the ROUGE-1 precision and recall using the standard F1-score formula.

> ROUGE-1 F1-score = 2 * (precision * recall) / (precision + recall) = 0.54

## ROUGE-2

Let’s try computing the ROUGE-2 considering 2-grams.

Remember our reference _R_ and candidate summary _C_:

- _R_: The cat is on the mat.
- _C_: The cat and the dog.

ROUGE-2 precision is the ratio of the number of 2-grams in _C_ that appear also in _R_ (only the 2-gram “the cat”), over the number of 2-grams in _C_.

> ROUGE-2 precision = 1/4 = 0.25

ROUGE-1 recall is the ratio of the number of 2-grams in _R_ that appear also in _C_ (only the 2-gram “the cat”), over the number of 2-grams in _R_.

> ROUGE-2 recall = 1/5 = 0.20

Therefore, the F1-score is:

> ROUGE-2 F1-score = 2 * (precision * recall) / (precision + recall) = 0.22

# ROUGE-L

**ROUGE-L** is based on the [longest common subsequence (LCS)](https://en.wikipedia.org/wiki/Longest_common_subsequence_problem) between our model output and reference, i.e. the longest sequence of words (not necessarily consecutive, but still in order) that is shared between both. A longer shared sequence should indicate more similarity between the two sequences.

We can compute ROUGE-L recall, precision, and F1-score just like we did with ROUGE-N, but this time we replace each n-gram match with the LCS.

Remember our reference _R_ and candidate summary _C_:

- _R_: The cat is on the mat.
- _C_: The cat and the dog.

The LCS is the 3-gram “the cat the” (remember that the words are not necessarily consecutive), which appears in both _R_ and _C_.

ROUGE-L precision is the ratio of the length of the LCS, over the number of unigrams in _C_.

> ROUGE-L precision = 3/5 = 0.6

ROUGE-L precision is the ratio of the length of the LCS, over the number of unigrams in _R_.

> ROUGE-L recall = 3/6 = 0.5

Therefore, the F1-score is:

> ROUGE-L F1-score = 2 * (precision * recall) / (precision + recall) = 0.55

# ROUGE-S

**ROUGE-S** allows us to add a degree of leniency to the n-gram matching performed with ROUGE-N and ROUGE-L. ROUGE-S is a skip-gram concurrence metric: this allows to search for consecutive words from the reference text that appear in the model output but are separated by one-or-more other words.

Consider the new reference _R_ and candidate summary _C_:

- _R_: The cat is on the mat.
- _C_: The gray cat and the dog.

If we consider the 2-gram “the cat”, the ROUGE-2 metric would match it only if it appears in _C_ exactly, but this is not the case since _C_ contains “the gray cat”. However, using ROUGE-S with unigram skipping, “the cat” would match “the gray cat” too.

We can compute ROUGE-S precision, recall, and F1-score in the same way as the other ROUGE metrics.

# Pros and Cons of ROUGE

This is the tradeoff to take into account when using ROUGE.

- _Pros_: it correlates positively with human evaluation, it’s inexpensive to compute and language-independent.
- _Cons_: ROUGE does not manage different words that have the same meaning, as it measures syntactical matches rather than semantics.

## ROUGE vs BLEU

In case you don’t know the BLEU metric already, I suggest that you read the companion article [Learn the BLEU metric by examples](https://medium.com/nlplanet/two-minutes-nlp-learn-the-bleu-metric-by-examples-df015ca73a86) to get a grasp on it.

In general:

- BLEU focuses on precision: how much the words (and/or n-grams) in the candidate model outputs appear in the human reference.
- ==ROUGE focuses on recall: how much the words (and/or n-grams) in the human references appear in the candidate model outputs.==

These results are complementing, as is often the case in the precision-recall tradeoff.