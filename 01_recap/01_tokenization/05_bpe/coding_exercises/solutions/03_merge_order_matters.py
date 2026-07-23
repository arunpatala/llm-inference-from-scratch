"""Solution to exercise 3. Verified output:

correct (training) order: ['ab', 'c']
reversed order:           ['a', 'bc']

Same two merge rules known in both cases, same input string "abc" -- but
reversing which one takes priority changes which pair merges first, which
changes the entire result. That's why encoding must replay the exact
learned order: the order encodes information (which pair was actually more
frequent in training), and throwing it away produces a tokenization that
doesn't match what the model was actually trained to expect.
"""


def encode(word, merge_priority):
    symbols = list(word)
    while True:
        pairs = [(symbols[i], symbols[i + 1]) for i in range(len(symbols) - 1)]
        candidates = [(merge_priority[p], i) for i, p in enumerate(pairs) if p in merge_priority]
        if not candidates:
            break
        _, i = min(candidates)
        symbols = symbols[:i] + ["".join(symbols[i:i + 2])] + symbols[i + 2:]
    return symbols


def main():
    merges = [("a", "b"), ("b", "c")]

    correct_priority = {pair: i for i, pair in enumerate(merges)}
    reversed_priority = {pair: i for i, pair in enumerate(reversed(merges))}

    word = "abc"
    print("correct (training) order:", encode(word, correct_priority))
    print("reversed order:          ", encode(word, reversed_priority))

    print()
    print("Same input, same two merge rules known, different order of")
    print("preference -- different output. That's the concrete answer to")
    print("why encoding must replay the exact learned order, not just any")
    print("order that happens to produce something valid.")


if __name__ == "__main__":
    main()
