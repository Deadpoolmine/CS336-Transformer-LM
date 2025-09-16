import os
from typing import BinaryIO
import regex as re
import multiprocessing
import cs336_basics.utils as utils

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_tokens: list[bytes],
) -> list[int]:
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
            for split_special_token in split_special_tokens:
                found_at = mini_chunk.find(split_special_token)
                if found_at != -1:
                    chunk_boundaries[bi] = initial_position + found_at
                    break

            if found_at != -1:
                break
    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def process_chunk(param):
    chunk_text, special_tokens = param
    pre_token_dict = {}

    chunks = re.split("|".join(map(re.escape, special_tokens)), chunk_text)
    # Run pre-tokenization on your chunk and store the counts for each pre-token
    pat = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    for chunk in chunks:
        ret = re.finditer(pat, chunk)
        for match in ret:
            token = match.group(0)
            pre_token_dict[token] = pre_token_dict.get(token, 0) + 1
    return pre_token_dict


def merge_one_pass(token, max_merge, merged_v):
    for elem_idx, elem in enumerate(token):
        if elem_idx + 1 < len(token):
            next_elem = token[elem_idx + 1]
            if (elem, next_elem) == max_merge:
                # replace the two elements with the merged value
                new_token = token[:elem_idx] + (merged_v,) + token[elem_idx + 2 :]
                return new_token, True
    return token, False

def bpe_debug_print(*args):
    utils.debug_print(verbose, *args)
        
def train_bpe(
    input_path: str, vocab_size: int, special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    merges = vocab_size - 256 - len(special_tokens)
    bpe_debug_print(input_path, vocab_size, special_tokens, merges)
    token_id = 0
    vocab = {}
    for st_id, st in enumerate(special_tokens):
        vocab[st_id] = bytes(st, "utf-8")
        token_id += 1

    for i in range(256):
        vocab[i + token_id] = bytes([i])
    token_id += 256

    ## Usage
    with open(input_path, "rb") as f:
        num_processes = 16
        boundaries = find_chunk_boundaries(
            f, num_processes, [bytes(st, "utf-8") for st in special_tokens]
        )

        # Example: process each chunk in parallel
        with multiprocessing.Pool(num_processes) as pool:
            f.seek(0)
            chunks = []
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                f.seek(start)
                chunk = f.read(end - start).decode("utf-8", errors="ignore")
                chunks.append((chunk, special_tokens))

            pre_token_dicts = pool.map(process_chunk, chunks)

    # Combine results from all chunks
    pre_token_dict = {}
    for token_dict in pre_token_dicts:
        for token, count in token_dict.items():
            # convert token to byte list
            token_bytes = tuple(token.encode("utf-8"))
            pre_token_dict[token_bytes] = pre_token_dict.get(token_bytes, 0) + count
    bpe_debug_print(pre_token_dict)

    reversed_merge_dict = {}
    merge_list = []

    def to_bytes(x):
        if x in reversed_merge_dict:
            assert x > 256
            return to_bytes(reversed_merge_dict[x][0]) + to_bytes(
                reversed_merge_dict[x][1]
            )
        else:
            assert x <= 256
            return bytes([x])

    # merge byte-pair for each token in pre_token_dict
    merge_dict = {}
    for token, count in pre_token_dict.items():
        for elem_idx, elem in enumerate(token):
            if elem_idx + 1 < len(token):
                next_elem = token[elem_idx + 1]
                merge = (elem, next_elem)
                merge_dict[merge] = merge_dict.get(merge, 0) + count

    for i in range(merges):
        # find all merges with the maximum count
        max_cnt = 0
        max_merges = []
        for merge, cnt in merge_dict.items():
            if cnt > max_cnt:
                max_cnt = cnt
                max_merges = [merge]
            elif cnt == max_cnt:
                max_merges.append(merge)

        # convert max_merges to 256 bytes and find the max
        format_max_merges = max_merges.copy()
        for idx, format_max_merge in enumerate(format_max_merges):
            a, b = format_max_merge
            format_max_merges[idx] = (to_bytes(a), to_bytes(b))
        max_format_merge = max(format_max_merges)
        max_merge = max_merges[format_max_merges.index(max_format_merge)]

        # mapping the merged bytes to a single new value (that exceeds 256)
        merged_v = 257 + i
        merge_list.append((max_merge, merged_v))
        reversed_merge_dict[merged_v] = max_merge

        bpe_debug_print(f"Merge {i+1}: try to merge {max_merge} into {merged_v}")
        
        # update pre_token_dict
        old_tokens = list(pre_token_dict.keys())
        for token in old_tokens:
            old_token = token
            new_token, merged = merge_one_pass(token, max_merge, merged_v)
            while merged:
                token = new_token
                new_token, merged = merge_one_pass(token, max_merge, merged_v)
            # replace the merged token in pre_token_dict
            if old_token != new_token:
                pre_token_dict[new_token] = (
                    pre_token_dict.get(new_token, 0) + pre_token_dict[old_token]
                )
                del pre_token_dict[old_token]
                
                bpe_debug_print(f"Merge {i+1}: merge {max_merge} into {merged_v}")
                bpe_debug_print(old_token, "->", new_token)
                # update merge_dict
                for elem_idx, elem in enumerate(new_token):
                    if elem == merged_v:
                        merge_dict[max_merge] -= pre_token_dict[new_token]
                        bpe_debug_print(f"Update {max_merge} ->{merge_dict[max_merge]}")
                        # check previous
                        if elem_idx - 1 >= 0:
                            prev_elem = new_token[elem_idx - 1]
                            merge = (prev_elem, merged_v)
                            merge_dict[merge] = (
                                merge_dict.get(merge, 0) + pre_token_dict[new_token]
                            )
                            bpe_debug_print(f"Update {merge} +{pre_token_dict[new_token]}")
                            
                            if prev_elem != merged_v:
                                orig_merge = (prev_elem, max_merge[0])
                                merge_dict[orig_merge] = (
                                    merge_dict.get(orig_merge, 0) - pre_token_dict[new_token]
                                )
                                bpe_debug_print(f"Update {orig_merge} -{pre_token_dict[new_token]}")
                        # check next
                        if elem_idx + 1 < len(new_token):
                            next_elem = new_token[elem_idx + 1]
                            if next_elem == merged_v:
                                # abab
                                merge = (max_merge[1], max_merge[0])
                                merge_dict[merge] = (
                                    merge_dict.get(merge, 0) - pre_token_dict[new_token]
                                )
                                bpe_debug_print(f"Update {merge} -{pre_token_dict[new_token]}")
                            else:
                                merge = (merged_v, next_elem)
                                merge_dict[merge] = (
                                    merge_dict.get(merge, 0) + pre_token_dict[new_token]
                                )
                                orig_merge = (max_merge[1], next_elem)
                                merge_dict[orig_merge] = (
                                    merge_dict.get(orig_merge, 0) - pre_token_dict[new_token]
                                )
                                bpe_debug_print(f"Update {merge} +{pre_token_dict[new_token]}")
                                bpe_debug_print(f"Update {orig_merge} -{pre_token_dict[new_token]}")
        
        # merge byte-pair for each token in pre_token_dict
        if golden:
            golden_merge_dict = {}
            for token, count in pre_token_dict.items():
                for elem_idx, elem in enumerate(token):
                    if elem_idx + 1 < len(token):
                        next_elem = token[elem_idx + 1]
                        merge = (elem, next_elem)
                        golden_merge_dict[merge] = golden_merge_dict.get(merge, 0) + count
            
            # compare merge_dict and golden_merge_dict
            error = False
            for merge in set(list(golden_merge_dict.keys())):
                if merge_dict.get(merge, 0) != golden_merge_dict.get(merge, 0):
                    bpe_debug_print(f"Error in merge_dict for {merge}: {merge_dict.get(merge, 0)} vs {golden_merge_dict.get(merge, 0)}")
                    error = True
            if error:
                exit(1)
    format_merge_list = []
    for merge in merge_list:
        (a, b), v = merge
        format_merge_list.append((to_bytes(a), to_bytes(b)))

    for id, format_merge in enumerate(format_merge_list):
        bpe_debug_print(f"id: {id}: {format_merge}")
        a, b = format_merge
        merged_bytes = a + b
        vocab[id + token_id] = merged_bytes

    # bpe_debug_print(vocab)
    # convert merge_list to the tuple[bytes, bytes]
    return vocab, format_merge_list

def load_vocab(vocab_filepath: str) -> dict[int, bytes]:
    vocab = {}
    # vocab_filepath is json file
    import json
    with open(vocab_filepath, "r", encoding="utf-8") as f:
        vocab_json: dict[str, bytes] = json.load(f)
        for k, v in vocab_json.items():
            k_encoded = k.encode("utf-8")
            vocab[int(v)] = bytes(k_encoded)
    return vocab

def load_merges(merges_filepath: str) -> list[tuple[bytes, bytes]]:
    merges = []
    with open(merges_filepath, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("#"):
                continue
            token1, token2 = line.strip().split()
            merges.append((bytes(token1, "utf-8"), bytes(token2, "utf-8")))
    return merges

verbose = False
golden = False

if __name__ == "__main__":
    train_bpe(
        "/home/deadpool/Playground/CS336/assignment1-basics/tests/fixtures/tinystories_sample_5M.txt",
        1000,
        ["<|endoftext|>"],
    )