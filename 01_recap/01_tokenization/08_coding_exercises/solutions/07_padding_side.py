"""Solution / answer to exercise 7. Verified output (Qwen/Qwen3-0.6B):

padding_side=right
input_ids:
tensor([[ 13048, 151643, 151643, 151643, 151643, 151643],
        [ 40451,    752,    264,   1293,   3364,    911]])
attention_mask:
tensor([[1, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1]])

padding_side=left
input_ids:
tensor([[151643, 151643, 151643, 151643, 151643,  13048],
        [ 40451,    752,    264,   1293,   3364,    911]])
attention_mask:
tensor([[0, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1]])

The answer: with right-padding, "Hi"'s one real token sits at position 0,
and positions 1-5 are all pad token 151643 -- so "take the last position's
hidden state" reads off a PAD token for this row, which is wrong. With
left-padding, the pad tokens come first and the real token always lands in
the actual last column, so "take the last position" is correct for every
row in the batch regardless of each prompt's original length. That's why
generation-time batching defaults to left-padding, while training-time
batching (which reads whole sequences, not just the last position)
conventionally uses right-padding.
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
