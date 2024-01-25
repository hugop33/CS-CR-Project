import torch
from torch import nn


class TransRLoss(nn.Module):
    """
    TransR loss function
    """
    def __init__(self):
        super(TransRLoss, self).__init__()

    def forward(self, true_plausibility, false_plausibility):
        """
        :param true_plausibility: plausibility score of positive samples
        :param false_plausibility: plausibility score of negative samples
        :return: loss value
        """
        loss = torch.sigmoid(false_plausibility - true_plausibility)
        return -torch.log(loss).sum()


if __name__ == '__main__':
    loss = TransRLoss()
    test_input1 = torch.ones((4, 1))
    test_input2 = torch.zeros((4, 1))
    loss_value = loss(test_input1, test_input2)
    print(loss_value, loss_value.shape)
