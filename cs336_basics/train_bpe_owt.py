from trainer import BPETrainer
from cs336_basics.common import gpt2_bytes_to_unicode

if __name__ == "__main__":
    train_file_path = "data/owt_train.txt"
    validation_file_path = "data/owt_valid.txt"
    trainer = BPETrainer(input_path=train_file_path, vocab_size=32000, special_tokens=["<|endoftext|>"], num_processes=16)
    vocab, merges = trainer.train()
    with open("vocab_owt.json", "w") as vocab_file:
        vocab_file.write(str(vocab))
    mapping = gpt2_bytes_to_unicode()
    with open("merges_owt.txt", "w") as merges_file:
        for merge in merges:
            merges_file.write((' '.join("".join(mapping[x] for x in word) for word in merge) + "\n"))


