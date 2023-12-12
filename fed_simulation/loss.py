# Author: Kyle Ma @ BCIL 
# Created: 04/26/2023
# Implementation of Automated Hematoma Segmentation

import torch
import torch.nn.functional as F

class DiceLoss(torch.nn.Module):
    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, predictions, targets):
    
        predictions = torch.sigmoid(predictions)
        
        num = predictions.size(0)
        
        predictions = predictions.view(num, -1)  # Flatten
        targets = targets.view(num, -1)  # Flatten
        
        intersection = torch.sum(predictions * targets)
        union = torch.sum(predictions) + torch.sum(targets)
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        loss = 1 - dice
        return loss
        
