[BLEU](https://aclanthology.org/P02-1040.pdf), or the Bilingual Evaluation Understudy, is a metric for comparing a candidate translation to one or more reference translations.

Although developed for translation, it can be used to evaluate text generated for different natural language processing tasks, such as paraphrasing and text summarization.

The BLEU score is not perfect, but it’s quick and inexpensive to calculate, language-independent, and, above all, correlates highly with human evaluation.

# How to compute the BLEU score

Consider the two reference translations _R1_ and _R2_ produced by human experts, and the candidate translation _C1_ produced by our translation system.

- _R1_: The cat is on the mat.
- _R2_: There is a cat on the mat.
- _C1_: The cat and the dog.

## Computing unigrams precision

To express the quality of our translation with a metric, we may count how many words in the candidate translation _C1_ are present in the reference translations _R1_ and _R2_, and divide the result by the number of words in _C1_ to get a percentage. Therefore, a perfect score is 1.0, whereas the worst score is 0.0. Let’s call this metric **BLEU***.

In _C1_ there are three words (_“the”, “cat”, “the”_) that appear on the reference translations, thus:

> BLEU*(C1) = 3/5 = 0.6

The candidate translation is far from perfect, indeed it receives a score of 0.6. Everything looks fine.

## The problem with repeating unigrams

Let’s compute the BLEU* score of the new candidate translation _C2_:

- _R1_: The cat is on the mat.
- _R2_: There is a cat on the mat.
- _C2_: The The The The The.

This time our translation system is not very good, unfortunately.

Every word in _C2_ is present in at least one between _R1_ and _R2_, thus:

> BLEU*(C2) = 5/5 = 1

We achieved a perfect score with a non-sense translation, there’s something we need to correct on our metric.

It doesn’t make sense to consider the word _“The”_ five times in the numerator, as it appears at most twice on each reference translation. We can try counting the word _“The”_ only for the times it appears at most on each reference translation, that is two. Let’s call this new metric **BLEU****.

> BLEU**(C2) = 2/5 = 0.4

Now the score makes more sense, as we are accounting for the fact that a good translated word appears too many times on our candidate translation.

## Considering n-grams

Let’s try computing the BLEU** score on two other candidate translations _C3_ and _C4_ to check if everything looks fine.

- _R1_: The cat is on the mat.
- _R2_: There is a cat on the mat.
- _C3_: There is a cat on the mat.
- _C4_: Mat the cat is on a there.

The BLEU** scores are the following:

> BLEU**(C3) = 7/7 = 1.0
> 
> BLEU**(C4) = 7/7 = 1.0

Both candidate translations contain words that are present in the reference translations, therefore they both achieve the maximum score. However, _C4_ is not a well-formed English sentence.

A quick way to get higher scores for well-formed sentences is to consider matching 2-grams or 3-grams instead of 1-grams only. Let’s call **BLEU**₁** the score that considers only 1-grams and **BLEU**₂** the score that considers only 2-grams.

_C3_ has six 2-grams and they all appear on the reference translation _R2_, thus:

> BLEU**₁(C3) = 7/7 = 1.0
> 
> BLEU**₂(C3) = 6/6 = 1.0

Instead, in _C4_ all the 2-grams don’t appear in any reference translation, thus:

> BLEU**₁(C4) = 7/7 = 1.0
> 
> BLEU**₂(C4) = 0/6 = 0.0

It is generally said that the **BLEU**ₙ** score for n-grams focuses on the sentence meaning for low _n_, and focuses on well-formed sentences for high _n_.

It has been found that the geometric mean of the BLEU**ₙ scores with _n_ between one and four has the best correlation with human evaluation, therefore it’s the score more commonly adopted. Let’s call it **MEAN_BLEU****.

## Penalizing short candidate translations

Let’s try now computing the BLEU**₁ and BLEU**₂ scores of the candidate translation _C5_:

- _R1_: The cat is on the mat.
- _R2_: There is a cat on the mat.
- _C5_: There is a cat.

The scores are:

> BLEU**₁(C5) = 4/4 = 1.0
> 
> BLEU**₂(C5) = 3/3= 1.0

Looks like _C5_ achieves a perfect BLEU**ₙ score for each _n_, even though the candidate translation is missing a piece of text with respect to the reference translations.

This can be avoided by adding a penalty for candidate translations whose length is less than the ones of the reference translations. We call it _Brevity Penalty (BP)_.

The final **BLEU** score is:

> BLEU = BP * MEAN_BLEU**

> ==That is, BLEU is the product of the Brevity Penalty BP (which penalizes short translations that don’t contain relevant text from the reference translations) and the geometric mean of the BLEU**ₙ scores for== ==_n_== ==between one and four (which takes into account small n-grams, to capture the sentence meaning, and large n-grams, to get well-formed sentences).==

What is the value of BP?

If the length of the candidate solution is bigger than the length of the reference translation with the most similar length, then we shouldn’t penalize and therefore BP equals one. Otherwise, BP is a decaying exponential which is lower when the length difference between the candidate and the reference translations is greater. The [BLEU paper](https://aclanthology.org/P02-1040.pdf) suggests computing the brevity penalty over the entire corpus rather than over single translations to smoothen the penalties for short translations.