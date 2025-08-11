from torch.nn import CrossEntropyLoss, MSELoss
import torch
import torch.nn as nn
class dist_softmax(nn.Module):
    def __init__(self, config):
        super(dist_softmax, self).__init__()
        self.entropyloss = CrossEntropyLoss()
        self.mseloss = MSELoss(reduction='mean')
        self.register_buffer('counter', torch.arange(config['output_dict_size']).float().to(config['device']))

    def forward(self, pred, target, alpha=0.1, beta=1.0):
        """
        Forward pass for the distance softmax loss.
        
        Args:
            pred (torch.Tensor): Predicted values.
            target (torch.Tensor): Target values.
            
        Returns:
            torch.Tensor: Computed loss value.
        """
        
        # Calculate the entropy loss
        entropyloss0 = self.entropyloss(pred, target).float()  # Cross-entropy loss between predicted and target distributions
        # Calculate the mean squared error loss
        pred_xy = nn.functional.softmax(pred, dim=-1)
        bins = torch.arange(pred_xy.size(-1), device=pred_xy.device).float()
        pred_xy = torch.sum(pred_xy * bins, dim=-1)  # Weighted average
        distance_loss = (pred_xy-target.float()).abs().mean()  # Mean squared error between predicted and target distances
        # Mean distance between predicted and target distances
        return entropyloss0 + alpha * distance_loss  # Combine the two losses with a weight factor

