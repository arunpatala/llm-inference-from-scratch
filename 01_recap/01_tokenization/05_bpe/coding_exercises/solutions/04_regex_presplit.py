"""Solution to exercise 4. Verified output:

without pre-split, first 5 merges: [('d', 'o'), ('do', 'g'), ('dog', '.'), ('c', 'a'), ('ca', 't')]
with pre-split, first 5 merges:    [('d', 'o'), ('do', 'g'), ('c', 'a'), ('ca', 't')]

Without pre-splitting, ('dog', '.') makes it into the merge list -- because
"dog." (with the period attached) was frequent enough in this corpus for
BPE to treat "dog" and "." as if they belong together. That wastes a vocab
slot on one specific punctuation combination, and it doesn't help at all
when the real text says "dog!" or "dog?" instead -- the model would need
to represent "dog" differently depending purely on what punctuation
happens to follow it.

With pre-splitting, "dog" and "." are never adjacent within the same
chunk, so that merge can never be proposed at all. "dog" collapses into a
single reusable token that works the same way regardless of what comes
after it.
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
    return re.findall(r"[A-Za-z]+|[^A-Za-z]+", word)


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
