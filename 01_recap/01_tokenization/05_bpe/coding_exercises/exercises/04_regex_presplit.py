"""
Exercise 4: what regex pre-splitting actually prevents.

Question 5: GPT-2/GPT-4-style tokenizers pre-split text by category
(letters, punctuation, ...) before BPE ever runs. Train the toy BPE trainer
(given below, same as exercise 1) on a corpus where "dog" is almost always
followed directly by a period, with and without pre-splitting, and watch
what merge appears -- or doesn't.

Run: python 04_regex_presplit.py
"""
import re
from collections import Counter

RAW_CORPUS = {"dog.": 20, "dog!": 1, "dog?": 1, "cat.": 5}


def get_pair_counts(word_freqs):
    pairs = Counter()
    for word, freq in word_freqs.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pairs[(symbols[i], symbols[i + 1])] += freq
    return pairs


def merge_pair(pair, word_freqs):
    bigram = " ".join(pair)
    merged = "".join(pair)
    return {word.replace(bigram, merged): freq for word, freq in word_freqs.items()}


def train_bpe(word_freqs, num_merges):
    word_freqs = dict(word_freqs)
    merges = []
    for _ in range(num_merges):
        pairs = get_pair_counts(word_freqs)
        if not pairs:
            break
        best_pair = max(pairs, key=pairs.get)
        merges.append(best_pair)
        word_freqs = merge_pair(best_pair, word_freqs)
    return merges


def presplit(word):
    """Split into letter-runs and non-letter-runs, so BPE never merges across the boundary."""
    # TODO: use re.findall with a pattern that matches either one-or-more
    # letters, or one-or-more non-letters, as separate chunks.
    return [word]


def main():
    no_presplit_corpus = {" ".join(list(word)): freq for word, freq in RAW_CORPUS.items()}
    merges_no_presplit = train_bpe(no_presplit_corpus, num_merges=5)
    print("without pre-split, first 5 merges:", merges_no_presplit)

    presplit_corpus = Counter()
    for word, freq in RAW_CORPUS.items():
        for chunk in presplit(word):
            presplit_corpus[" ".join(list(chunk))] += freq
    merges_presplit = train_bpe(dict(presplit_corpus), num_merges=5)
    print("with pre-split, first 5 merges:   ", merges_presplit)

    print()
    print("Look for ('dog', '.') in one list and not the other. What would")
    print("go wrong at inference time if that merge existed, given that")
    print("'dog!' and 'dog?' are also real, valid continuations?")


if __name__ == "__main__":
    main()
