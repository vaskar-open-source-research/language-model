import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from src.language_model.transformer_block import TransformerBlock
from src.language_model.rmsnorm import RMSNorm
from src.language_model.softmax import softmax

class KVCache:

    def __init__(self):
        self.keys = None
        self.values = None

    def extend(self, key, value):
        if self.keys is None:
            self.keys = key
            self.values = value
        else:
            self.keys = torch.cat([self.keys, key], dim=-2)
            self.values = torch.cat([self.values, value], dim=-2)
        return self.keys, self.values

    def length(self):
        if self.keys is None:
            return 0
        return self.keys.shape[-2]


class TransformerLM(nn.Module):

    def __init__(self, vocab_size, context_length, d_model, num_layers, num_heads, d_ff, attn_pdrop, residual_pdrop, device='cuda'):
        super(TransformerLM, self).__init__()
        self.residual_pdrop = residual_pdrop
        self.token_embeddings = nn.Embedding(vocab_size, d_model).to(device)
        self.position_embeddings = nn.Embedding(context_length, d_model).to(device)
        self.layers = nn.Sequential(
            OrderedDict([
                (f'{i}', TransformerBlock(d_model, num_heads, d_ff, attn_pdrop, residual_pdrop).to(device)) for i in range(num_layers)
            ])
        )
        self.ln_final = RMSNorm(d_model).to(device)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False).to(device)
        self.device = device
    
    def load_state_dict(self, weights):
        self.token_embeddings.weight = nn.Parameter(weights['token_embeddings.weight'])
        self.position_embeddings.weight = nn.Parameter(weights['position_embeddings.weight'])
        for layer in range(len(self.layers)):
            self.layers[layer].load_state_dict({
                'attn.q_proj.weight' : weights[f'layers.{layer}.attn.q_proj.weight' if f'layers.{layer}.attn.q_proj.weight' in weights else f'layers.{layer}.attn.q_proj'],
                'attn.k_proj.weight' : weights[f'layers.{layer}.attn.k_proj.weight' if f'layers.{layer}.attn.k_proj.weight' in weights else f'layers.{layer}.attn.k_proj'],
                'attn.v_proj.weight' : weights[f'layers.{layer}.attn.v_proj.weight' if f'layers.{layer}.attn.v_proj.weight' in weights else f'layers.{layer}.attn.v_proj'],
                'attn.output_proj.weight' : weights[f'layers.{layer}.attn.output_proj.weight' if f'layers.{layer}.attn.output_proj.weight' in weights else f'layers.{layer}.attn.output_proj'],
                'ffn.w1.weight': weights[f'layers.{layer}.ffn.w1.weight'],
                'ffn.w2.weight': weights[f'layers.{layer}.ffn.w2.weight'],
                'ln1.weight': weights[f'layers.{layer}.ln1.weight'],
                'ln2.weight': weights[f'layers.{layer}.ln2.weight'],
            })
        self.ln_final.load_state_dict({'weight': weights['ln_final.weight']})
        self.lm_head.weight = nn.Parameter(weights['lm_head.weight'])

    def forward(self, x, return_hidden_states=False, cache=None):
        bs, seq_len = x.shape
        token_emb = self.token_embeddings(x)
        if cache is not None:
            offset_seq_len = cache[0].length()
        else:
            offset_seq_len = 0
        pos_emb = self.position_embeddings(torch.stack([torch.arange(seq_len, device=self.device) + offset_seq_len] * bs))
        x = F.dropout(token_emb + pos_emb, self.residual_pdrop)
        for i, layer in enumerate(self.layers):
            if cache is not None:
                x = layer(x, cache[i])
            else:
                x = layer(x)
            if i == len(self.layers) - 1:
                final_hidden_states = x
        logits = self.lm_head(self.ln_final(x))
        
        if return_hidden_states:
            return logits, final_hidden_states
        else:
            return logits
    
    def generate(self, x, max_new_tokens, temperature, do_sample, eos_token, pad_token, cache=None):
        eos_mask = torch.zeros(x.shape[0], device=self.device)
        full_text = x
        for _ in range(max_new_tokens):
            logits = self.forward(x, cache=cache)
            if do_sample:
                probs = softmax(logits[:, -1, :] / temperature)
                next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)
                next_token = torch.where(eos_mask == 1, pad_token, next_token)
                eos_mask = torch.where(next_token == eos_token, torch.ones_like(eos_mask), eos_mask)
            else:
                next_token = torch.argmax(logits[:, -1, :], dim=-1)
                next_token = torch.where(eos_mask == 1, pad_token, next_token)
                eos_mask = torch.where(next_token == eos_token, torch.ones_like(eos_mask), eos_mask)

            if cache is None:
                x = torch.cat([x, next_token[:, None]], dim=-1)
            else:
                x = next_token[:, None]
            full_text = torch.cat([full_text, next_token[:, None]], dim=-1)

        return full_text
    
    def print_device(self):
        print(f"Device: {self.device}")

if __name__ == "__main__":
    from src.tokenizer import Tokenizer
    import time
    model = TransformerLM(vocab_size=10001, context_length=256, d_model=512, num_layers=4, num_heads=16, d_ff=2048, attn_pdrop=0.0, residual_pdrop=0.0, device='cuda')
    ckpt_path = "/mnt/efs/vaskarnath/practice/spring2024-assignment1-basics/ckpt/checkpoint_157000.pt"
    vocab_path = "/mnt/efs/vaskarnath/practice/spring2024-assignment1-basics/cs336_basics/data/TinyStoriesV2-GPT4-valid_vocab.json"
    merges_path = "/mnt/efs/vaskarnath/practice/spring2024-assignment1-basics/cs336_basics/data/TinyStoriesV2-GPT4-valid_merges.txt"
    tokenizer = Tokenizer.from_files(vocab_path, merges_path, ["<|endoftext|>"])
    model.load_state_dict(torch.load(ckpt_path)['model'])

    text = "Once "
    input_ids = tokenizer.encode(text)
    print(f"Text: {tokenizer.decode(input_ids.tolist())}")
    c = [KVCache() for _ in range(4)]
    batch_size = 100
    model.eval()
    torch.manual_seed(0)
    cache_start_time = time.time()
    with torch.no_grad():
        output_ids_cache = model.generate(
            input_ids.unsqueeze(0).repeat(batch_size, 1).to(model.device), 
            max_new_tokens=250, 
            temperature=0.7, 
            do_sample=False, 
            eos_token=tokenizer.eos_token_id, 
            pad_token=tokenizer.pad_token_id, 
            cache=c
        )
    cache_end_time = time.time()
    torch.manual_seed(0)
    no_cache_start_time = time.time()
    with torch.no_grad():
        output_ids = model.generate(
            input_ids.unsqueeze(0).repeat(batch_size, 1).to(model.device), 
            max_new_tokens=250, 
            temperature=0.7, 
            do_sample=False, 
            eos_token=tokenizer.eos_token_id, 
            pad_token=tokenizer.pad_token_id, 
            cache=None
        )
    no_cache_end_time = time.time()
    assert torch.allclose(output_ids_cache, output_ids), f"Output ids are not the same, {output_ids_cache} != {output_ids}"
    
    print(f"Time taken with cache: {cache_end_time - cache_start_time} seconds")
    print(f"Time taken without cache: {no_cache_end_time - no_cache_start_time} seconds")
