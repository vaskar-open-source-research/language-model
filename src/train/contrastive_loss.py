import torch
import torch.nn.functional as F

def contrastive_loss(hidden_states, n_samples: int):
    """
    hidden_states: [batch_size, context_length, d_model]
    """
    batch_size = hidden_states.shape[0]
    context_length = hidden_states.shape[1]
    d_model = hidden_states.shape[2]

    device = hidden_states.device
    
    # for each sample in the batch, sample n_samples of source and future tokens
    source_indices = torch.randint(0, context_length, (batch_size, n_samples), device=device)
    future_indices = torch.randint(0, context_length, (batch_size, n_samples), device=device)

    # get the hidden states for the source and future tokens
    source_hidden_states = torch.gather(hidden_states, dim=1, index=source_indices.unsqueeze(2).expand(-1, -1, d_model))
    future_hidden_states = torch.gather(hidden_states, dim=1, index=future_indices.unsqueeze(2).expand(-1, -1, d_model))

    # reshape to have n_samples as the first dimension
    source_hidden_states = source_hidden_states.permute(1, 0, 2) # [n_samples, batch_size, d_model]
    future_hidden_states = future_hidden_states.permute(1, 0, 2) # [n_samples, batch_size, d_model]

    # matrix multiplication between the hidden states and the target token an clamp by abs 20
    # einsum_logits = torch.einsum('nbs,nds->nbd', source_hidden_states, future_hidden_states) # [n_samples, batch_size, batch_size]
    logits = torch.matmul(source_hidden_states, future_hidden_states.transpose(1, 2)) # [n_samples, batch_size, batch_size]
    logits = torch.clamp(logits, -20, 20)

    # apply sigmoid bce loss between the dot product and diagonal matrix of 1s and 0s
    target = torch.stack([torch.eye(batch_size, device=device) for _ in range(n_samples)]) # [n_samples, batch_size, batch_size]
    loss = F.binary_cross_entropy_with_logits(logits, target)
    
    # return the mean of the loss
    return loss.mean()

if __name__ == "__main__":
    # test the contrastive loss
    batch_size, context_length, d_model = 2, 4, 8
    hidden_states = torch.randn(batch_size, context_length, d_model)
    print(contrastive_loss(hidden_states, 1))
 