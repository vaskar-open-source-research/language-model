import regex as re

def corpus_to_pre_tokens(corpus: str):
    PAT = r"""'s|'t|'re|'ve|'m|'ll|'d| ?[\p{L}]+| ?[\p{N}]+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    pre_tokens = re.findall(PAT, corpus)
    return pre_tokens