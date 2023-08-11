import torch
from torch import nn


class CRF(nn.Module):
    def __init__(self, num_nodes, iteration=10):
        super(CRF, self).__init__()
        self.num_nodes = num_nodes
        self.iteration = iteration
        self.W = nn.Parameter(torch.zeros(1, num_nodes, num_nodes))

    def forward(self, feats, logits):
        # Iterate over the batch
        Q_list = []
        for batch_idx in range(feats.shape[0]):
            feats_single = feats[batch_idx]
            logits_single = logits[batch_idx]
            feats_norm = torch.norm(feats_single, p=2, dim=-1, keepdim=True)
            pairwise_norm = feats_norm * feats_norm.transpose(-1, 0)
            pairwise_dot = feats_single @ feats_single.transpose(-1, 0)

            pairwise_sim = pairwise_dot / pairwise_norm
            W_sym = (self.W + torch.transpose(self.W, 1, 2)) / 2
            pairwise_potential = pairwise_sim * W_sym.squeeze()
            unary_potential = logits_single.clone()

            # Initialize the approximate distribution Q
            Q = torch.zeros_like(logits_single).fill_(0.5)

            for i in range(self.iteration):
                # Update Q
                Q = torch.sigmoid(unary_potential + torch.sum(pairwise_potential.unsqueeze(-1) * Q, dim=1, keepdim=True))

            Q_list.append(Q.unsqueeze(0))

        return torch.cat(Q_list, dim=0)

    def __repr__(self):
        return 'CRF(num_nodes={}, iteration={})'.format(
            self.num_nodes, self.iteration
        )



