"""Solution to exercise 1. Verified output:

merge 1: ('e', 's') (count=9)
merge 2: ('es', 't') (count=9)
merge 3: ('est', '</w>') (count=9)
merge 4: ('l', 'o') (count=7)
merge 5: ('lo', 'w') (count=7)
merge 6: ('n', 'e') (count=6)
merge 7: ('ne', 'w') (count=6)
merge 8: ('new', 'est</w>') (count=6)
merge 9: ('low', '</w>') (count=5)
merge 10: ('w', 'i') (count=3)

final word forms:
  'low</w>' (freq 5)
  'low e r </w>' (freq 2)
  'newest</w>' (freq 6)
  'wi d est</w>' (freq 3)

'low' and 'newest' fully collapsed into single symbols because they were
frequent enough to earn every possible merge. 'lower' and 'widest' didn't
fully collapse -- 'e r' and 'd' never got merged because ('e','r') and
similar pairs never became the single most-frequent pair anywhere in the
corpus before the merge budget (10) ran out.
"""
from collections import Counter

CORPUS = {
    "l o w </w>": 5,
    "l o w e r </w>": 2,
    "n e w e s t </w>": 6,
    "w i d e s t </w>": 3,
}


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
