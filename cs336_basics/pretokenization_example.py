import os
from typing import BinaryIO
import regex as re
import multiprocessing

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

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
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def process_chunk(chunk_text):
    pre_token_dict = {}
    # Run pre-tokenization on your chunk and store the counts for each pre-token
    pat = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    ret = re.finditer(pat, chunk_text)
    for match in ret:
        token = match.group(0)
        pre_token_dict[token] = pre_token_dict.get(token, 0) + 1
    return pre_token_dict

## Usage
with open("/home/deadpool/Playground/CS336/data/TinyStoriesV2-GPT4-valid.txt", "rb") as f:
    num_processes = 16
    boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
    
    # Example: process each chunk in parallel
    with multiprocessing.Pool(num_processes) as pool:
        f.seek(0)
        chunks = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            chunks.append(chunk)
            
        pre_token_dicts = pool.map(process_chunk, chunks)

    # Combine results from all chunks
    pre_token_dict = {}
    for token_dict in pre_token_dicts:
        for token, count in token_dict.items():
            # convert token to byte list
            token_bytes = tuple(token.encode("utf-8"))
            pre_token_dict[token_bytes] = pre_token_dict.get(token_bytes, 0) + count
    print(pre_token_dict)
    
    merges = 10
    merge_list = []
    
    for i in range(merges):
        merge_dict = {}
        for token, count in pre_token_dict.items():
            for elem_idx, elem in enumerate(token):
                if elem_idx + 1 < len(token):
                    next_elem = token[elem_idx + 1]
                    merge = (elem, next_elem)
                    merge_dict[merge] = merge_dict.get(merge, 0) + count
        max_merge = max(merge_dict, key=merge_dict.get)
        # get the maximized size of bytes
        merged_v = 257 + i
        merge_list.append((max_merge, merged_v))
        print(f"Merge {i+1}: try to merge {max_merge} into {merged_v}")
        
        # replace the token in pre_token_dict
        old_tokens = list(pre_token_dict.keys())
        for token in old_tokens:
            def merge_one_pass(token, max_merge, merged_v):
                # token can be [a, b, x, e]
                for elem_idx, elem in enumerate(token):
                    if elem_idx + 1 < len(token):
                        next_elem = token[elem_idx + 1]
                        if (elem, next_elem) == max_merge:
                            # replace the two elements with the merged value
                            new_token = token[:elem_idx] + (merged_v,) + token[elem_idx + 2:]
                            return new_token, True
                return token, False
            
            old_token = token
            new_token, merged = merge_one_pass(token, max_merge, merged_v)
            while merged:
                token = new_token
                new_token, merged = merge_one_pass(token, max_merge, merged_v)
            
            if old_token != new_token:
                pre_token_dict[new_token] = pre_token_dict.get(new_token, 0) + pre_token_dict[old_token]
                del pre_token_dict[old_token]