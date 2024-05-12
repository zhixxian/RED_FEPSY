import torch
import torch.nn as nn
import torch.nn.functional as F

class RankingLoss(nn.Module):
    def __init__(self, num_classes: int = 10,
                       margin: float = 0.1,
                       alpha: float = 0.1,
                       ignore_index: int =-100):
        super().__init__()
        self.margin = margin
        self.alpha = alpha
        self.ignore_index = ignore_index
        self.cross_entropy = nn.CrossEntropyLoss()

    @property
    def names(self):
        return "loss"

    def get_logit_diff(self, inputs, mixup):
        max_values, indices = inputs.max(dim=1)
        max_values = max_values.unsqueeze(dim=1)
       
        max_values_mixup, indices_mixup = mixup.max(dim=1)
        max_values_mixup = max_values_mixup.unsqueeze(dim=1)
        # diff = max_values - max_values_mixup
        diff = max_values_mixup -  max_values 

        return diff
    
    def get_conf_diff(self, inputs, mixup):
        inputs = F.softmax(inputs, dim=1)
        max_values, indices = inputs.max(dim=1)
        max_values = max_values.unsqueeze(dim=1)

        mixup = F.softmax(mixup, dim=1)
        max_values_mixup, indices_mixup = mixup.max(dim=1)
        max_values_mixup = max_values_mixup.unsqueeze(dim=1)
        
        # diff = max_values - max_values_mixup
        diff = max_values_mixup -  max_values 

        return diff

    def forward(self, inputs, targets, mixup, target_re, lam):
        if inputs.dim() > 2:
            inputs = inputs.view(inputs.size(0), inputs.size(1), -1)  # N,C,H,W => N,C,H*W
            inputs = inputs.transpose(1, 2)    # N,C,H*W => N,H*W,C
            inputs = inputs.contiguous().view(-1, inputs.size(2))   # N,H*W,C => N*H*W,C
            targets = targets.view(-1)

        if self.ignore_index >= 0:
            index = torch.nonzero(targets != self.ignore_index).squeeze()
            inputs = inputs[index, :]
            targets = targets[index]

        loss_ce = self.cross_entropy(inputs, targets)
        
        self_mixup_mask = (target_re == 1.0).sum(dim=1).reshape(1, -1) 
        self_mixup_mask = (self_mixup_mask.sum(dim=0) == 0.0) 
     
        # diff = self.get_conf_diff(inputs, mixup) # using probability
        diff = self.get_logit_diff(inputs, mixup)
        loss_mixup = (self_mixup_mask * F.relu(diff+self.margin)).mean()

        loss = loss_ce + self.alpha * loss_mixup

        return loss
