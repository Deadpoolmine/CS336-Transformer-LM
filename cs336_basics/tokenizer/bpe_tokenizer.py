import pstats
from typing import Iterable, Iterator
import cs336_basics.tokenizer.bpe_train as bpe_train
import cs336_basics.utils as utils
import regex as re
import cProfile

verbose = False


def tokenizer_debug_print(*args):
    utils.debug_print(verbose, *args)


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.vocab = vocab
        self.merges = merges
        self.merge_search = {pair: idx for idx, pair in enumerate(merges)}
        self.special_tokens = special_tokens if special_tokens is not None else []
        # sort special tokens
        self.special_tokens.sort(key=lambda x: -len(x))

        self.reserve_vocab = {}
        for id, token in vocab.items():
            self.reserve_vocab[token] = id

        # for merge in self.merges:
        #     tokenizer_debug_print(merge)
        # for vocab_id, vocab_bytes in self.vocab.items():
        #     tokenizer_debug_print(f"{vocab_id}: {vocab_bytes}")

    @staticmethod
    def from_files(
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,
    ):
        vocab = bpe_train.load_vocab(vocab_filepath)
        merges = bpe_train.load_merges(merges_filepath)
        return Tokenizer(vocab, merges, special_tokens)

    def __try_merge_bytes(
        self, byte1: bytes, byte2: bytes
    ) -> tuple[bytes | None, int | None]:
        pair = (byte1, byte2)
        if pair in self.merge_search:
            return byte1 + byte2, self.merge_search[pair]
        return None, None

    # merge only two tokens in one pass, feels like very deficient
    def __merge_one_pass(
        self, cur_token_bytes: list[bytes]
    ) -> tuple[list[bytes], bool]:
        new_token_bytes = []
        merge_list = []
        for byte_idx, cur_byte in enumerate(cur_token_bytes):
            if byte_idx == len(cur_token_bytes) - 1:
                break

            next_byte = cur_token_bytes[byte_idx + 1]
            # convert int to bytes
            cur_byte = utils.try_convert_to_bytes(cur_byte)
            next_byte = utils.try_convert_to_bytes(next_byte)
            merged, merged_idx = self.__try_merge_bytes(cur_byte, next_byte)
            if merged is not None:
                merge_list.append((byte_idx, merged, merged_idx))

        priority = min(merge_list, key=lambda x: x[2]) if len(merge_list) > 0 else None
        min_merge = priority

        if min_merge is not None:
            new_token_bytes = (
                cur_token_bytes[: min_merge[0]]
                + [min_merge[1]]
                + cur_token_bytes[min_merge[0] + 2 :]
            )
            is_merged = True
        else:
            new_token_bytes = cur_token_bytes
            is_merged = False

        return new_token_bytes, is_merged

    def __encode_one_pre_token(self, pre_token: str) -> list[int]:
        results = []
        if pre_token in self.special_tokens:
            token_bytes = pre_token.encode("utf-8")
            token_id = self.reserve_vocab.get(token_bytes, -1)
            if token_id == -1:
                raise ValueError(f"Special token {pre_token} not in vocab.")
            results.append(token_id)
            return results

        pre_token_bytes = pre_token.encode("utf-8")

        # it is a binary-tree like merge
        max_merge_cnt = utils.round_up_to_power_of_2(len(pre_token_bytes)) - 1

        cur_token_bytes = list(pre_token_bytes)
        tokenizer_debug_print(cur_token_bytes)
        tokenizer_debug_print(f"Merging {pre_token_bytes}...")
        for i in range(max_merge_cnt):
            tokenizer_debug_print(f"Merge pass {i + 1}")
            cur_token_bytes, is_merged = self.__merge_one_pass(cur_token_bytes)
            if not is_merged:
                break

        tokenizer_debug_print(cur_token_bytes)
        tokenizer_debug_print()

        # retrieve token ids from vocab
        for token_bytes in cur_token_bytes:
            token_bytes = utils.try_convert_to_bytes(token_bytes)
            token_id = self.reserve_vocab.get(token_bytes, -1)
            if token_id == -1:
                raise ValueError(f"Token {token_bytes} not in vocab.")
            results.append(token_id)
        return results

    def __process_text(self, text: str) -> list[str]:
        pre_tokens = []
        special_tokens_in_text = []

        if len(self.special_tokens) == 0:
            text_blks = [text]
        else:
            text_blks = re.split("|".join(map(re.escape, self.special_tokens)), text)
            special_tokens_in_text = re.findall(
                "|".join(map(re.escape, self.special_tokens)), text
            )
            tokenizer_debug_print(special_tokens_in_text)

        # Run pre-tokenization on your chunk and store the counts for each pre-token
        pat = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        for text_blk in text_blks:
            ret = re.finditer(pat, text_blk)
            for match in ret:
                token = match.group(0)
                pre_tokens.append(token)
            if len(special_tokens_in_text) > 0:
                pre_tokens.append(special_tokens_in_text.pop(0))

        tokenizer_debug_print(pre_tokens)
        return pre_tokens
    
    def encode(self, text: str) -> list[int]:
        results = []
        pre_tokens = self.__process_text(text)
        # merge bytes for each pre_token
        for pre_token in pre_tokens:
            _results = self.__encode_one_pre_token(pre_token)
            results.extend(_results)

        return results

    # find the complete matched token that ends before target_pos
    # thanks to ChatGPT and Copilot
    def __find_safe_split_position(self, text: str, target_pos: int) -> int:
        if target_pos >= len(text):
            return len(text)

        search_start = max(0, target_pos - target_pos // 4)
        search_text = text[search_start : target_pos + target_pos // 4]

        # find the token boundaries in search_text
        pat = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        matches = list(re.finditer(pat, search_text))

        if not matches:
            return 0

        # find the best match that ends before or at target_pos
        best_pos = 0
        for match in matches:
            absolute_end = search_start + match.end()
            if absolute_end <= target_pos:
                best_pos = absolute_end
            else:
                break

        return best_pos

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        buffer = ""
        chunk_size = 4096 # 4096 bytes

        for idx, text in enumerate(iterable):
            buffer += text
            while len(buffer) > chunk_size:
                # find safe split position
                safe_split_pos = self.__find_safe_split_position(buffer, chunk_size)

                if safe_split_pos > 0:
                    chunk_to_process = buffer[:safe_split_pos]
                    encoded = self.encode(chunk_to_process)
                    # cProfile.runctx('encoded = self.encode(chunk_to_process)', globals(), locals(), "stats")
                    # p = pstats.Stats("stats")
                    # p.sort_stats('time').print_stats(40)
                    for token_id in encoded:
                        yield token_id

                    # update buffer pointer
                    buffer = buffer[safe_split_pos:]
                else:
                    # add warn
                    raise Warning("A single token is too long to fit in chunk size.")
                    break

        # handle remaining buffer
        if buffer:
            encoded = self.encode(buffer)
            for token_id in encoded:
                yield token_id

    def decode(self, ids: list[int]) -> str:
        decoded_bytes = bytearray()
        for id in ids:
            if id in self.vocab:
                decoded_bytes.extend(self.vocab[id])
            else:
                decoded_bytes.extend(bytes[id])

        return decoded_bytes.decode("utf-8", errors="replace")


if __name__ == "__main__":
    vocab, merges = bpe_train.train_bpe(
        "/home/deadpool/Playground/CS336/data/test.txt",
        256 + 8,
        ["<|endoftext|>"],
    )

    tokenizer = Tokenizer(vocab, merges, ["<|endoftext|>"])
    # text = "abab <|endoftext|>"
    text = "🙃"
    encoded = tokenizer.encode(text)
    tokenizer_debug_print("Encoded:", encoded)
    # decoded = tokenizer.decode(encoded)
    # tokenizer_debug_print("Decoded:", decoded)
