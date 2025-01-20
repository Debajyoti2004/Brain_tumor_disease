import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    def __init__(self, smooth=1):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        
    def forward(self, preds, targets):
        preds = preds.contiguous()
        targets = targets.contiguous()
        
        intersection = (preds * targets).sum(dim=(2, 3))
        union = (preds.sum(dim=(2, 3)) + targets.sum(dim=(2, 3)))
        
        dice = 2 * (intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()

class ComboLoss(nn.Module):
    def __init__(self, smooth=1, alpha=0.5):
        super(ComboLoss, self).__init__()
        self.alpha = alpha
        self.dice = DiceLoss(smooth=smooth)

        self.bce_with_logits = nn.BCEWithLogitsLoss()

    def forward(self, preds, targets):
        bc_loss = self.bce_with_logits(preds, targets)
        dice_loss = self.dice(preds, targets)

        combo = self.alpha * bc_loss + (1 - self.alpha) * dice_loss
        return combo

if __name__ == "__main__":
    batch_size, height, width = 4, 96, 128
    preds = torch.randn(batch_size, 1, height, width)  # Raw logits, no sigmoid applied
    targets = torch.randint(0, 2, (batch_size, 1, height, width)).float()  

    combo_loss = ComboLoss(smooth=1, alpha=0.5)
    loss = combo_loss(preds, targets)
    print(f"Combo Loss: {loss.item()}")
