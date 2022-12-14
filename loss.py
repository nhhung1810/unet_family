from torch import Tensor, nn


class DiceLoss(nn.Module):

    def __init__(self):
        super(DiceLoss, self).__init__()
        self.smooth = 1.0

    def forward(self, y_pred: Tensor, y_true: Tensor):
        assert y_pred.size() == y_true.size(), f"Un-match size: y_pred=={y_pred.size()} but y_true=={y_true.size()} "
        
        y_pred = y_pred[:, 0].contiguous().view(-1)
        y_true = y_true[:, 0].contiguous().view(-1)
        
        intersection = (y_pred * y_true).sum()
        dsc = (2. * intersection +
               self.smooth) / (y_pred.sum() + y_true.sum() + self.smooth)
        return 1. - dsc