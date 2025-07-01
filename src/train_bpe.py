import argparse
import regex as re
from typing import List
from collections import Counter
import json
import os
from tqdm import tqdm
from src.utils import corpus_to_pre_tokens
from sortedcontainers import SortedSet
from heapq import heappush, heappop 


def naive_train_bpe(input_path: str, vocab_size: int, special_tokens: List[str]):
    
    with open(input_path, 'r') as fp:
        corpus = fp.read()

    pre_tokens = corpus_to_pre_tokens(corpus)
    pre_token_bytes = [tuple([bytes([c]) for c in s.encode('utf-8')]) for s in pre_tokens]
    vocab = {
        i: i.to_bytes() for i in range(256)
    }
    for special_token in special_tokens:
        vocab[len(vocab)] = special_token.encode('utf-8')

    merged_bytes = []

    while pre_token_bytes and len(vocab) < vocab_size:
        pre_token_bytes_counter = Counter(pre_token_bytes)
        byte_pair_counts = {}
        for token_bytes, value in pre_token_bytes_counter.items():
            for i in range(1, len(token_bytes)):
                byte_pair = (token_bytes[i-1], token_bytes[i])
                if byte_pair not in byte_pair_counts:
                    byte_pair_counts[byte_pair] = 0
                byte_pair_counts[byte_pair] += value
        
        most_freqent_byte_pair = sorted(list(byte_pair_counts.items()), key=lambda x: (x[1], x[0]))[-1][0]
        merged_bytes.append(most_freqent_byte_pair)
        vocab[len(vocab)] = most_freqent_byte_pair[0] + most_freqent_byte_pair[1]

        if len(vocab) == vocab_size:
            break
        # merge the bytes in pre_token_bytes
        new_pre_token_bytes = []
        for i, token_bytes in enumerate(pre_token_bytes):
            new_token_bytes = []
            i = 0
            while i < len(token_bytes):
                if i + 1 < len(token_bytes) and (token_bytes[i], token_bytes[i+1]) == most_freqent_byte_pair:
                    new_token_bytes.append(token_bytes[i] + token_bytes[i+1])
                    i += 2
                else:
                    new_token_bytes.append(token_bytes[i])
                    i += 1
            if len(new_token_bytes) > 1:
                new_pre_token_bytes.append(tuple(new_token_bytes))
        pre_token_bytes = new_pre_token_bytes

    return vocab, merged_bytes

def get_pre_tokens(text: str, special_tokens: List[str], show_progress: bool = False):
    if special_tokens:
        
        min_heap = []
        
        for special_token in tqdm(special_tokens, disable=not show_progress):
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
    
    return pre_tokens

def optmized_train_bpe(input_path: str, vocab_size: int, special_tokens: List[str], verbose: bool = False):
    
    if verbose:
        print(f"Started training...")
    if verbose:
        print(f"Started reading file...")
    with open(input_path, 'r') as fp:
        # handle reading large files
        corpus = ""
        for line in tqdm(fp, disable=not verbose, desc="Reading file"):
            corpus += line
    if verbose:
        print(f"Finished reading file.")

    if verbose:
        print(f"Generating pre tokens...")
    pre_tokens = get_pre_tokens(corpus, special_tokens, verbose)
    if verbose:
        print(f"Generated pre tokens.")

    pre_token_bytes = [tuple([bytes([c]) for c in s.encode('utf-8')]) for s in tqdm(pre_tokens, disable=not verbose, desc="Encoding pre tokens")]
    if verbose:
        print(f"Encoded pre tokens.")

    vocab = {
        i: bytes([i]) for i in range(256)
    }

    seen_bytes = set()
    for special_token in special_tokens:
        vocab[len(vocab)] = special_token.encode('utf-8')
        seen_bytes.add(special_token.encode('utf-8'))
    
    merged_bytes = []
    
    if verbose:
        print(f"Counting pre tokens...")
    pre_token_bytes_counter = Counter(pre_token_bytes)
    list_of_pre_token_bytes_counter = list([seq, cnt] for seq, cnt in pre_token_bytes_counter.items())
    if verbose:
        print(f"Counted pre tokens.")

    byte_pair_counts = {}
    byte_pair_indices = {}
    
    if verbose:
        print(f"Generating byte pairs...")
    for k, (token_bytes, value) in tqdm(enumerate(list_of_pre_token_bytes_counter), disable=not verbose):
        for i in range(1, len(token_bytes)):
            byte_pair = (token_bytes[i-1], token_bytes[i])
            if byte_pair not in byte_pair_counts:
                byte_pair_counts[byte_pair] = 0
            byte_pair_counts[byte_pair] += value
            if byte_pair not in byte_pair_indices:
                    byte_pair_indices[byte_pair] = []
            if i < len(token_bytes):
                byte_pair_indices[byte_pair].append(k)
    if verbose:
        print(f"Generated byte pairs...")
    
    byte_pair_counts_set = SortedSet((v, k[0], k[1]) for k, v in byte_pair_counts.items())

    with tqdm(total=vocab_size - len(vocab), desc="Train", disable=not verbose) as pbar:
        while pre_token_bytes and len(vocab) < vocab_size:

            pbar.update(1)
            # most_freqent_byte_pair = sorted(list(byte_pair_counts.items()), key=lambda x: (x[1], x[0]))
            # find the first byte pair from the end of the list that is not in vocab
            # for i in range(len(byte_pair_counts) - 1, -1, -1):
            #     byte_pair = byte_pair_counts.peekitem(i)[0]
            #     if byte_pair[0] + byte_pair[1] not in seen_bytes:
            #         most_freqent_byte_pair = byte_pair
            #         break
                # else:
                #     byte_pair_counts.popitem(byte_pair)
                #     byte_pair_indices.popitem(byte_pair)
            while True:
                most_freqent_byte_pair = byte_pair_counts_set.pop(-1)
                if most_freqent_byte_pair[1] + most_freqent_byte_pair[2] not in seen_bytes:
                    break
            most_freqent_byte_pair = (most_freqent_byte_pair[1], most_freqent_byte_pair[2])
            merged_bytes.append(most_freqent_byte_pair)
            vocab[len(vocab)] = most_freqent_byte_pair[0] + most_freqent_byte_pair[1]
            seen_bytes.add(most_freqent_byte_pair[0] + most_freqent_byte_pair[1])

            if len(vocab) == vocab_size:
                break
            # merge the bytes in pre_token_bytes
            byte_pair_counts.pop(most_freqent_byte_pair)
            
            for idx in byte_pair_indices[most_freqent_byte_pair]:
                seq, count = list_of_pre_token_bytes_counter[idx]
                new_token_bytes = []
                i = 0
                while i < len(seq):
                    if i + 1 < len(seq) and (seq[i], seq[i+1]) == most_freqent_byte_pair:
                        new_token_bytes.append(seq[i] + seq[i+1])
                        
                        if i - 1 >= 0 and (seq[i-1], seq[i]) in byte_pair_counts:
                            byte_pair_counts_set.remove((byte_pair_counts[(seq[i-1], seq[i])], seq[i-1], seq[i]))
                            byte_pair_counts[(seq[i-1], seq[i])] -= count
                            byte_pair_counts_set.add((byte_pair_counts[(seq[i-1], seq[i])], seq[i-1], seq[i]))
                        
                        if i + 2 < len(seq) and (seq[i+1], seq[i+2]) in byte_pair_counts:
                            byte_pair_counts_set.remove((byte_pair_counts[(seq[i+1], seq[i+2])], seq[i+1], seq[i+2]))
                            byte_pair_counts[(seq[i+1], seq[i+2])] -= count
                            byte_pair_counts_set.add((byte_pair_counts[(seq[i+1], seq[i+2])], seq[i+1], seq[i+2]))
                        
                        if i - 1 >= 0:
                            left = (seq[i - 1], seq[i] + seq[i+1])
                            if left not in byte_pair_counts:
                                byte_pair_counts_set.add((0, left[0], left[1]))
                                byte_pair_counts[left] = 0
                            if left not in byte_pair_indices:
                                byte_pair_indices[left] = []
                            byte_pair_indices[left].append(idx)
                            byte_pair_counts_set.remove((byte_pair_counts[left], left[0], left[1]))
                            byte_pair_counts[left] += count
                            byte_pair_counts_set.add((byte_pair_counts[left], left[0], left[1]))

                        if i + 2 < len(seq):
                            right = (seq[i] + seq[i + 1], seq[i + 2])
                            if right not in byte_pair_counts:
                                byte_pair_counts_set.add((0, right[0], right[1]))
                                byte_pair_counts[right] = 0
                            if right not in byte_pair_indices:
                                byte_pair_indices[right] = []
                            byte_pair_indices[right].append(idx)
                            byte_pair_counts_set.remove((byte_pair_counts[right], right[0], right[1]))
                            byte_pair_counts[right] += count
                            byte_pair_counts_set.add((byte_pair_counts[right], right[0], right[1]))
                        i += 2
                    else:
                        new_token_bytes.append(seq[i])
                        i += 1
                
                list_of_pre_token_bytes_counter[idx][0] = new_token_bytes
            
    return vocab, merged_bytes


def main():

    # example call: python cs336_basics/train_bpe.py --input_path cs336_basics/test_files/test_corpus.txt --vocab_size 300 --special_tokens hello
    parser = argparse.ArgumentParser(description='Train bpe tokenizer.')
    parser.add_argument("--input_path", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--vocab_size", type=int)
    parser.add_argument("--special_tokens", nargs='+', type=str)
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()
        
    vocab, merges = optmized_train_bpe(args.input_path, args.vocab_size, args.special_tokens, args.verbose)

    dir_name = os.path.dirname(args.input_path)
    file_name = args.input_path.split('/')[-1].split('.')[0]

    # save in the subdirectory {file_name}_tokenizer/
    tokenizer_dir = args.output_dir
    os.makedirs(tokenizer_dir, exist_ok=True)

    with open(os.path.join(tokenizer_dir, f'vocab.json'), 'w') as fp:
        vocab_dict = {}
        for int_key, byte_value in vocab.items():
            if ''.join([chr(x) for x in list(byte_value)]) in vocab_dict:
                print(f"Duplicate key: {''.join([chr(x) for x in list(byte_value)])}")
                print(f"byte_value: {byte_value}")
                print(f"int_key: {int_key}")
            vocab_dict[''.join([chr(x) for x in list(byte_value)])] = int_key
        json.dump(vocab_dict, fp)
        
    with open(os.path.join(tokenizer_dir, f'merges.txt'), 'w') as fp:
        for i, merge in enumerate(merges):
            merge_first = ''.join([chr(x) for x in list(merge[0])]).replace(' ', 'Ġ').replace('\n', 'Ċ')
            merge_second = ''.join([chr(x) for x in list(merge[1])]).replace(' ', 'Ġ').replace('\n', 'Ċ')
            if i == len(merges) - 1:
                fp.write(f"{merge_first} {merge_second}")
            else:
                fp.write(f"{merge_first} {merge_second}\n")
    
if __name__ == "__main__":
    main()
