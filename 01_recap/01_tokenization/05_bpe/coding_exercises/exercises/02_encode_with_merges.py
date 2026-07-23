"""
Exercise 2: encode brand-new text using a frozen merge list.

Question 4 (../01_questions.md): encoding doesn't recount frequencies, it
replays the learned merges in order. The merge list below is exercise 1's
real output. Implement the encoder, then encode "lowest" -- a word that
never appeared whole in training, only as pieces of "low" and "newest".

Run: python 02_encode_with_merges.py
"""

# Real output from exercise 1, training on the Sennrich et al. (2016) toy corpus
MERGES = [
    ("e", "s"), ("es", "t"), ("est", "</w>"), ("l", "o"), ("lo", "w"),
    ("n", "e"), ("ne", "w"), ("new", "est</w>"), ("low", "</w>"), ("w", "i"),
]
MERGE_PRIORITY = {pair: i for i, pair in enumerate(MERGES)}


def encode(word, merge_priority):
    """Apply the learned merges to `word`, highest-priority (earliest-learned) merge first."""
    symbols = list(word) + ["</w>"]
    while True:
        pairs = [(symbols[i], symbols[i + 1]) for i in range(len(symbols) - 1)]
        # TODO: among the pairs that appear in merge_priority, find the one
        # with the LOWEST priority index (= learned earliest = highest
        # priority), merge it, and continue. If none of the current pairs
        # are in merge_priority, stop and return `symbols`.
        break
    return symbols


def main():
    for word in ["low", "lowest", "newer", "widest", "slow"]:
        print(f"{word!r:>10} -> {encode(word, MERGE_PRIORITY)}")

    print()
    print("'lowest' was never a whole training word -- only 'low' and")
    print("'newest' were. Does it still encode sensibly? What does that")
    print("tell you about BPE generalizing to unseen combinations?")


if __name__ == "__main__":
    main()
