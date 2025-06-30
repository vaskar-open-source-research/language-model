import torch
import os

def save_checkpoint(model, optimizer, iteration, out):
    state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'iteration': iteration
    }
    os.makedirs(os.path.dirname(out), exist_ok=True)
    torch.save(state, out)

def load_checkpoint(model, optimizer, out):
    checkpoint = torch.load(out)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    iteration = checkpoint['iteration']
    return iteration

if __name__ == "__main__":
    from cs336_basics.language_model.transformer_lm import TransformerLM
    from cs336_basics.train.adamw import AdamW
    ckpt_path = "/mnt/efs/vaskarnath/practice/spring2024-assignment1-basics/ckpt/checkpoint_8.pt"
    model = TransformerLM(vocab_size=10000, context_length=1024, d_model=128, num_layers=1, num_heads=2, d_ff=128, attn_pdrop=0.1, residual_pdrop=0.1, device='cuda')
    optimizer = AdamW(model.parameters(), weight_decay=0.01, lr=0.0001)
    load_checkpoint(model, optimizer, ckpt_path)
    print(f"model: {model}")
    print(f"optimizer: {optimizer}")

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"model parameters: {count_parameters(model)}")
