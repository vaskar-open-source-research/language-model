from typing import Iterable, List, Dict, Tuple, Iterator
import json
from src.utils import corpus_to_pre_tokens
import copy
from tqdm import tqdm
from heapq import heappush, heappop
import torch
from argparse import ArgumentParser
import os

class Tokenizer:

    def __init__(self, vocab: Dict[int, bytes], merges: List[Tuple[bytes, bytes]], special_tokens: list[str] | None = None, pad_token: str = '<|pad|>', eos_token: str = '<|endoftext|>'):
        self.vocab = vocab
        self.rvocab = {
            value: key for key, value in self.vocab.items()
        }
        self.merges = merges
        if special_tokens:
            self.special_tokens = sorted(special_tokens, key=lambda x: -len(x))
        else:
            self.special_tokens = None
        
        if special_tokens:
            for special_token in special_tokens:
                special_token_bytes = special_token.encode('utf-8')
                if special_token_bytes not in self.vocab.values():
                    self.vocab[len(self.vocab)] = special_token_bytes

        # self.pad_token = pad_token
        # self.eos_token = eos_token
        
        # pad_token_bytes = pad_token.encode('utf-8')
        # if pad_token_bytes not in self.vocab.values():
        #     key = len(self.vocab)
        #     self.vocab[key] = pad_token_bytes
        #     self.rvocab[pad_token_bytes] = key
        
        # eos_token_bytes = eos_token.encode('utf-8')
        # if eos_token_bytes not in self.vocab.values():
        #     key = len(self.vocab)
        #     self.vocab[key] = eos_token_bytes
        #     self.rvocab[eos_token_bytes] = key
        
        # self.pad_token_id = self.rvocab[pad_token_bytes]
        # self.eos_token_id = self.rvocab[eos_token_bytes]

        # if self.pad_token not in self.special_tokens:
        #     self.special_tokens.append(self.pad_token)
        # if self.eos_token not in self.special_tokens:
        #     self.special_tokens.append(self.eos_token)

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None):
        with open(vocab_filepath, 'r') as fp:
            vocab = json.load(fp)
        
        vocab_dict = {}

        for byte_key_str, int_value in vocab.items():
            vocab_dict[int_value] = bytes([ord(x) for x in list(byte_key_str)])

        merges = []
        with open(merges_filepath, 'r') as fp:
            for line in fp.readlines():
                s1, s2 = line.split(' ')
                merges.append(tuple([bytes([ord(x) for x in s1.replace('Ġ', ' ').replace('Ċ', '\n')]), bytes([ord(x) for x in s2.replace('Ġ', ' ').replace('Ċ', '\n')])]))
        
        return cls(vocab_dict, merges, special_tokens)
        
    def encode(self, text: str, return_tensors: bool = False, show_progress: bool = False) -> List[int] | torch.Tensor:
        if self.special_tokens:
            
            min_heap = []
            
            for special_token in self.special_tokens:
                prev_idx = 0
                idx = text.find(special_token, prev_idx)
                while idx != -1:
                    heappush(min_heap, (idx, -len(special_token)))
                    prev_idx = idx + len(special_token)
                    idx = text.find(special_token, prev_idx)
            
            split_strings = []
            special_token_mask = []
            i = 0
            prev_idx = 0
            
            while min_heap:
                idx, length = heappop(min_heap)
                length = -length
                if idx < prev_idx:
                    continue
                if len(text[prev_idx:idx]):
                    split_strings.append(text[prev_idx:idx])
                    special_token_mask.append(0)
                split_strings.append(text[idx:idx + length])
                special_token_mask.append(1)
                prev_idx = idx + length
                        
            if len(text[prev_idx:len(text)]):
                split_strings.append(text[prev_idx:len(text)])
                special_token_mask.append(0)
            
            pre_tokens = []
            for i in tqdm(range(len(split_strings)), total=len(split_strings), desc="Splitting text with special tokens", disable=not show_progress):
                s, mask = split_strings[i], special_token_mask[i]
                if mask:
                    pre_tokens.append(s)
                else:
                    pre_tokens.extend(corpus_to_pre_tokens(s))
        else:
            pre_tokens = corpus_to_pre_tokens(text)

        pre_token_bytes = set()
        for pre_token in tqdm(pre_tokens, total=len(pre_tokens), desc="Text to bytes", disable=not show_progress):
            pre_token_byte = pre_token.encode('utf-8')
            if pre_token_byte in self.rvocab:
                pre_token_bytes.add((pre_token, pre_token_byte))
            else:
                pre_token_bytes.add((pre_token, tuple([bytes([c]) for c in pre_token_byte])))

        merges_to_position = {}
        for i, merge in enumerate(self.merges):
            merges_to_position[merge] = i

        pre_token_to_ids_mp = {}
        for pre_token, pre_token_byte in tqdm(pre_token_bytes, total=len(pre_token_bytes), desc="Encoding text", disable=not show_progress):
            pre_token_full_byte = pre_token.encode('utf-8')
            if pre_token_full_byte in self.rvocab:
                pre_token_to_ids_mp[pre_token] = [self.rvocab[pre_token_full_byte]]
                continue
            
            curr_pre_token_byte = list(pre_token_byte)

            while len(curr_pre_token_byte) > 2:
                minimum_merge_position = float('inf')
                minimum_merge_index = None
                for i in range(len(curr_pre_token_byte) - 1):
                    if (curr_pre_token_byte[i], curr_pre_token_byte[i + 1]) in self.merges:
                        if merges_to_position[(curr_pre_token_byte[i], curr_pre_token_byte[i + 1])] < minimum_merge_position:
                            minimum_merge_position = merges_to_position[(curr_pre_token_byte[i], curr_pre_token_byte[i + 1])]
                            minimum_merge_index = i
                if minimum_merge_index is not None:
                    curr_pre_token_byte = curr_pre_token_byte[:minimum_merge_index] + [curr_pre_token_byte[minimum_merge_index] + curr_pre_token_byte[minimum_merge_index + 1]] + curr_pre_token_byte[minimum_merge_index + 2:]
                else:
                    break
            
            curr_token_ids = []
            for token_byte in curr_pre_token_byte:
                curr_token_ids.append(self.rvocab[token_byte])
            pre_token_to_ids_mp[pre_token] = curr_token_ids

        token_ids = []
        for pre_token in pre_tokens:
            token_ids.extend(pre_token_to_ids_mp[pre_token])

        if return_tensors:
            return torch.tensor(token_ids)
        else:   
            return token_ids

    def encode_iterable(self, iterable: Iterable[str], return_tensors: bool = False, show_progress: bool = False) -> List[int] | torch.Tensor:
        token_ids = []
        for text in iterable:
            token_ids.extend(self.encode(text, return_tensors=return_tensors, show_progress=show_progress))
        return token_ids

    def decode(self, ids: list[int]) -> str:
        if len(ids) == 0:
            return ''
        all_bytes = None
        for id in ids:
            if all_bytes is None:
                all_bytes = self.vocab[id]
            else:
                all_bytes += self.vocab[id]
        return all_bytes.decode('utf-8', errors='replace')


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--tokenizer_path", type=str)
    parser.add_argument("--input_path", type=str)
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--show_progress", action="store_true")
    args = parser.parse_args()

    tokenizer = Tokenizer.from_files(os.path.join(args.tokenizer_path, 'vocab.json'), os.path.join(args.tokenizer_path, 'merges.txt'))
    with open(args.input_path, 'r') as fp:
        corpus = fp.read()
        token_ids = tokenizer.encode(corpus, return_tensors=True, show_progress=args.show_progress)
        torch.save(token_ids, args.output_path)