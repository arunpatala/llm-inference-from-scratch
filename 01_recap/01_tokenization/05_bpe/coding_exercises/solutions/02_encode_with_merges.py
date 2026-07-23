"""Solution to exercise 2. Verified output:

     'low' -> ['low</w>']
  'lowest' -> ['low', 'est</w>']
   'newer' -> ['new', 'e', 'r', '</w>']
  'widest' -> ['wi', 'd', 'est</w>']
    'slow' -> ['s', 'low</w>']

'lowest' encodes as ['low', 'est</w>'] -- reusing the 'low' merge learned
from "low"/"lower" and the 'est</w>' merge learned from "newest", even
though "lowest" itself never appeared in training. That's BPE
generalizing: merges are learned from substrings, not whole words, so they
recombine onto words the trainer never saw.

'slow' shows the other side of it: 's' stays a lone, unmerged character,
because ('s', 'l') was never a candidate pair anywhere in this tiny corpus.
"""

MERGES = [
    ("e", "s"), ("es", "t"), ("est", "</w>"), ("l", "o"), ("lo", "w"),
    ("n", "e"), ("ne", "w"), ("new", "est</w>"), ("low", "</w>"), ("w", "i"),
]
MERGE_PRIORITY = {pair: i for i, pair in enumerate(MERGES)}


def encode(word, merge_priority):
    symbols = list(word) + ["</w>"]
    while True:
        pairs = [(symbols[i], symbols[i + 1]) for i in range(len(symbols) - 1)]
        candidates = [(merge_priority[p], i) for i, p in enumerate(pairs) if p in merge_priority]
        if not candidates:
            break
        _, i = min(candidates)
        symbols = symbols[:i] + ["".join(symbols[i:i + 2])] + symbols[i + 2:]
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
