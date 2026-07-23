"""
Exercise 1: implement the BPE training loop.

Question 2 (../01_questions.md) asked you to describe this algorithm in
words. Now implement it: given word frequencies (each word already split
into space-separated symbols, with a `</w>` end-of-word marker), repeatedly
find the most frequent adjacent symbol pair and merge it, recording each
merge in order.

The corpus below is the classic toy example from Sennrich, Haddow, Birch
(2016), Table 1 -- the paper that introduced BPE for this purpose.

Run: python 01_train_toy_bpe.py
"""
from collections import Counter

CORPUS = {
    "l o w </w>": 5,
    "l o w e r </w>": 2,
    "n e w e s t </w>": 6,
    "w i d e s t </w>": 3,
}


def get_pair_counts(word_freqs):
    """Count how often each adjacent symbol pair occurs, weighted by word frequency."""
    # TODO: for each word, split it into symbols, then for every adjacent
    # pair of symbols add `freq` to that pair's count in a Counter.
    pairs = Counter()
    return pairs


def merge_pair(pair, word_freqs):
    """Replace every occurrence of `pair` (as adjacent symbols) with the merged symbol."""
    # TODO: for each word, replace the space-separated bigram "x y" with the
    # merged symbol "xy" wherever it occurs. Return a new dict.
    return dict(word_freqs)


def train_bpe(word_freqs, num_merges):
    word_freqs = dict(word_freqs)
    merges = []
    for step in range(num_merges):
        pairs = get_pair_counts(word_freqs)
        if not pairs:
            break
        best_pair = max(pairs, key=pairs.get)
        merges.append(best_pair)
        print(f"merge {step + 1}: {best_pair} (count={pairs[best_pair]})")
        word_freqs = merge_pair(best_pair, word_freqs)
    return merges, word_freqs


def main():
    merges, final = train_bpe(CORPUS, num_merges=10)

    print()
    print("final word forms:")
    for word, freq in final.items():
        print(f"  {word!r} (freq {freq})")

    print()
    print("merge list, in the order they were learned:")
    print(merges)


if __name__ == "__main__":
    main()
