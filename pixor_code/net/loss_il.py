import torch.nn.functional as F
import torch.nn as nn
import torch


### Loss
class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.epsilon = 1e-7
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, predict, target):
        # assert target.size() == predict.size(), f'{target.size()} != {predict.size()}'

        pt_1 = torch.where(target == 1, predict, torch.ones_like(predict))
        pt_0 = torch.where(target == 0, predict, torch.zeros_like(predict))

        # Clip to prevent NaN's and Inf's
        pt_1 = torch.clamp(pt_1, self.epsilon, 1.0 - self.epsilon)
        pt_0 = torch.clamp(pt_0, self.epsilon, 1.0 - self.epsilon)

        pt_1_sum = torch.sum(self.alpha * torch.pow(1 - pt_1, self.gamma) * torch.log(pt_1))
        pt_0_sum = torch.sum((1 - self.alpha) * torch.pow(pt_0, self.gamma) * torch.log(1 - pt_0))

        return -(pt_1_sum + pt_0_sum)


class CustomLoss(nn.Module):
    def __init__(self, num_classes=1):
        super(CustomLoss, self).__init__()
        self.num_classes = num_classes
        self.focal_loss = FocalLoss()

    def forward(self, cls_pred, reg_pred, cls_target, reg_target):
        batch_size = reg_target.size(0)

        cls_loss = self.focal_loss(cls_pred, cls_target)
        reg_loss = F.l1_loss(reg_pred, reg_target, reduction='sum')  # TODO: try mse
        #         reg_loss = F.mse_loss(reg_pred, reg_target, reduction='sum')  # TODO: try learn where class > 0

        cls_loss = cls_loss / batch_size
        reg_loss = reg_loss / batch_size

        return cls_loss, reg_loss
