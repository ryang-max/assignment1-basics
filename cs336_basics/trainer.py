from collections import defaultdict
import os
from typing import BinaryIO
import regex as re
import multiprocessing as mp
from tqdm import tqdm


PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
NUM_PROCESSES = 2  # Number of processes to use for parallel processing

class BPETrainer:
    def __init__(self, input_path: str, vocab_size: int, special_tokens: list[str] = None):
        self.input_path = input_path
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens if special_tokens else []
        self.PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    def find_chunk_boundaries(
            self,
        file: BinaryIO, 
        desired_num_chunks: int = 8, 
        split_special_token: bytes = "<|endoftext|>".encode("utf-8")
    ):
        """
        Chunk the file into parts that can be counted independently.
        May return fewer chunks if the boundaries end up overlapping.
        """
        assert isinstance(split_special_token, bytes), (
            "Must represent special token as a bytestring"
        )

        # Get total file size in bytes
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)

        chunk_size = file_size // desired_num_chunks

        # Initial guesses for chunk boundary locations, uniformly spaced
        # Chunks start on previous index, don't include last index
        chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
        chunk_boundaries[-1] = file_size

        mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

        for bi in range(1, len(chunk_boundaries) - 1):
            initial_position = chunk_boundaries[bi]
            file.seek(initial_position)  # Start at boundary guess
            while True:
                mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

                # If EOF, this boundary should be at the end of the file
                if mini_chunk == b"":
                    chunk_boundaries[bi] = file_size
                    break

                # Find the special token in the mini chunk
                found_at = mini_chunk.find(split_special_token)
                if found_at != -1:
                    chunk_boundaries[bi] = initial_position + found_at + len(split_special_token)
                    break
                initial_position += mini_chunk_size

        # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
        return sorted(set(chunk_boundaries))

    def _pretokenize(self, start: int, end: int) -> dict[str, int]:
        """
        Tokenizes the input text using a regex pattern.
        """
        tokens = defaultdict(int)
        with open(self.input_path, "r") as f:
            f.seek(start)
            text = f.read(end - start)
            articles = re.split("|".join([re.escape(token) for token in self.special_tokens]), text)
            for article in articles:
                for item in re.finditer(self.PAT, article):
                    tokens[item.group()] += 1
        return tokens
    
    def pretokenize(self):
        chunk_boundaries = self.find_chunk_boundaries(
            open(self.input_path, "rb"),
            desired_num_chunks=NUM_PROCESSES
        )
        # print(f"Chunk boundaries: {chunk_boundaries}")
        all_tokens = defaultdict(int)
        with mp.Pool(processes=len(chunk_boundaries) - 1) as pool:
            async_results = [pool.apply_async(
                self._pretokenize, args=(chunk_boundaries[i], chunk_boundaries[i + 1]))
                for i in range(len(chunk_boundaries) - 1)
            ]
            results = [res.get() for res in async_results]
            for result in results:
                for token, count in result.items():
                    all_tokens[token.encode("utf-8")] += count
        #return all_tokens
        return [([bytes([each]) for each in k], v) for k, v in all_tokens.items()]

    def train(self):
        all_tokens = self.pretokenize()
        vocab = defaultdict(int)
        merges = list()
        idx = 0
        for token in self.special_tokens:
            vocab[idx] = token.encode("utf-8")
            idx += 1
        for i in range(256):
            vocab[idx] = bytes([i])
            idx += 1
        while len(vocab) < self.vocab_size:
            # Find the most frequent pair of tokens
            pairs = defaultdict(int)
            for token, count in all_tokens:
                if len(token) < 2:
                    continue
                for i in range(len(token) - 1):
                    pair = (token[i], token[i + 1])
                    pairs[pair] += count
            selected_pair = max(pairs.items(), key=lambda x: (x[1], x[0][0], x[0][1]))
            # print(f"Selected pair: {selected_pair}")
            merges.append(selected_pair[0])
            new_token = selected_pair[0][0] + selected_pair[0][1]
            vocab[idx] = new_token
            idx += 1
            # Update all tokens with the new merged token
            new_all_tokens = []
            for token, count in all_tokens:
                new_token_list = []
                i = 0
                while i < len(token):
                    if i < len(token) - 1 and (token[i], token[i + 1]) == selected_pair[0]:
                        new_token_list.append(new_token)
                        i += 2
                    else:
                        new_token_list.append(token[i])
                        i += 1
                new_all_tokens.append((new_token_list, count))
            all_tokens = new_all_tokens
        return vocab, merges
            

if __name__ == "__main__":
    file_path = "tests/fixtures/tinystories_sample.txt"
    trainer = BPETrainer(input_path=file_path, vocab_size=300, special_tokens=["<|endoftext|>"])
    print(trainer.train())
    #another_trainer = BPETrainer(input_path=file_path, vocab_size=1000, special_tokens=["<endoftext|>"])
            #print(re.split("|".join([re.escape(token) for token in special_tokens]), s))
