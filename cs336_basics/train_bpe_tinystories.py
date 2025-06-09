from trainer import BPETrainer
from cs336_basics.common import gpt2_bytes_to_unicode
import json

if __name__ == "__main__":
    # train_file_path = "tests/fixtures/tinystories_sample_5M.txt"
    train_file_path = "data/TinyStoriesV2-GPT4-train.txt"
    validation_file_path = "data/TinyStoriesV2-GPT4-valid.txt"
    trainer = BPETrainer(input_path=train_file_path, vocab_size=10000, special_tokens=["<|endoftext|>"], num_processes=64)
    vocab, merges = trainer.train()
    with open("vocab_tinystory.json", "w") as vocab_file:
        json.dump()
    mapping = gpt2_bytes_to_unicode()
    with open("merges_tinystory.txt", "w") as merges_file:
        for merge in merges:
            merges_file.write((' '.join("".join(mapping[x] for x in word) for word in merge) + "\n"))


