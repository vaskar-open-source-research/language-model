import math
import torch
import numpy as np

# def data_loader(dataset,  batch_size: int, context_length: int, device: str): 
#     indices = np.arange(len(dataset) - context_length)
#     sampled_indices = np.random.choice(indices, size=batch_size, replace=False)
#     dataset = np.stack([dataset for _ in range(batch_size)], axis=0)
#     indices = [np.arange(sampled_index, sampled_index + context_length + 1) for sampled_index in sampled_indices]
#     dataset_indices = np.stack([dataset[i, indices[i]] for i in range(batch_size)], axis=0)
#     input_ids = torch.from_numpy(dataset_indices[:,:-1]).to(device)
#     labels = torch.from_numpy(dataset_indices[:,1:]).to(device)
#     return input_ids, labels

def data_loader(dataset, batch_size: int, context_length: int, device: str = 'cuda'): 
    if isinstance(dataset, torch.Tensor):
        dataset = dataset.to(device)
    else:
        dataset = torch.tensor(dataset).to(device)
    range = len(dataset) - context_length
    rng = np.random.default_rng()          # fastest generator in NumPy ≥1.17
    sampled = rng.choice(range, batch_size, replace=False)
    batch_indices = sampled[:, None] + np.arange(context_length + 1)[None, :] # advanced indexing
    batch_data = dataset[batch_indices]
    input_ids = batch_data[:, :-1].to(device)
    labels = batch_data[:, 1:].to(device)
    return input_ids, labels