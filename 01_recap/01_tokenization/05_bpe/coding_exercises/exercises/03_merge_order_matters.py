"""
Exercise 3: merge order isn't arbitrary -- it changes the result.

Question 4 asked why encoding has to replay merges in the exact order they
were learned, rather than any order that produces *a* valid result. Here's
a minimal case where reversing the order genuinely changes the output, not
just the process.

Suppose training saw "ab" more often than "bc", so BPE learned merge
("a","b") before ("b","c"). Encode "abc" with the real order, then with the
order reversed, using the encode() function from exercise 2 (reproduced
here, already correct).

Run: python 03_merge_order_matters.py
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

    # TODO: build two priority dicts from `merges` -- one in the real,
    # learned order, one with the order reversed. (Hint: enumerate() over
    # `merges` vs. over `reversed(merges)`.)
    correct_priority = None
    reversed_priority = None

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
