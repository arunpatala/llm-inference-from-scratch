"""
Exercise 7: padding side actually matters.

Question 18 raised this: padding side (left vs. right) affects correctness
when batching prompts of different lengths, not just cosmetics. Nothing to
implement here -- run it, look at both outputs, and reason through the
question at the bottom yourself before checking ../../solutions.

Run: python 07_padding_side.py
"""
from transformers import AutoTokenizer

MODEL_ID = "Qwen/Qwen3-0.6B"


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    prompts = ["Hi", "Tell me a long story about"]

    for side in ["right", "left"]:
        tokenizer.padding_side = side
        batch = tokenizer(prompts, padding=True, return_tensors="pt")
        print(f"padding_side={side}")
        print("input_ids:")
        print(batch["input_ids"])
        print("attention_mask:")
        print(batch["attention_mask"])
        print()

    print("A causal LM predicts the next token from the LAST position's")
    print("hidden state. For the short prompt 'Hi', which padding side puts")
    print("the real last token in the actual last position of the row --")
    print("and which one leaves a pad token sitting there instead?")


if __name__ == "__main__":
    main()
