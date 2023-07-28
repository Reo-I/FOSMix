import numpy as np
import torch
import torch.nn as nn


def _threshold(x, threshold=None):
    if threshold is not None:
        return (x > threshold).type(x.dtype)
    else:
        return x


def iou(pr, gt, eps=1e-7, threshold=None, for_metric = None):
    """Calculate Intersection over Union between ground truth and prediction
    Args:
        pr (torch.Tensor): predicted tensor
        gt (torch.Tensor):  ground truth tensor
        eps (float): epsilon to avoid zero division
        threshold: threshold for outputs binarization
    Returns:
        float: IoU (Jaccard) score
    """

    pr = _threshold(pr, threshold=threshold)
    pr = pr.type(gt.dtype)
    intersection = torch.sum((gt * pr).float())
    union = torch.sum(gt) + torch.sum(pr) - intersection + eps
    if for_metric:
        # intersection == 0 and (union-eps)==0 ->iou = 1.0になってしまう
        return (intersection) / union 
    else:
        return (intersection + eps) / union


def fscore(pr, gt, beta=1, eps=1e-7, threshold=None):
    """Calculate F-score between ground truth and prediction
    Args:
        pr (torch.Tensor): predicted tensor
        gt (torch.Tensor):  ground truth tensor
        beta (float): positive constant
        eps (float): epsilon to avoid zero division
        threshold: threshold for outputs binarization
    Returns:
        float: F score
    """

    pr = _threshold(pr, threshold=threshold)
    tp = torch.sum(gt * pr)
    fp = torch.sum(pr) - tp
    fn = torch.sum(gt) - tp
    score = ((1 + beta ** 2) * tp + eps) / (
        (1 + beta ** 2) * tp + beta ** 2 * fn + fp + eps
    )
    return score


class Fscore(nn.Module):
    def __init__(self, class_weights=1.0, threshold=None):
        super().__init__()
        self.class_weights = torch.tensor(class_weights)
        self.threshold = threshold
        self.name = "Fscore"

    @torch.no_grad()
    def forward(self, input, target):
        input = torch.softmax(input, dim=1).argmax(dim=1)
        scores = []
        for i in range(1, input.shape[1]):  # background is not included
            ypr = input[:, i, :, :].sigmoid()
            ygt = target[:, i, :, :]
            scores.append(fscore(ypr, ygt, threshold=self.threshold))
        return sum(scores) / len(scores)


class IoU(nn.Module):
    def __init__(self, class_weights=1.0, threshold=None):
        super().__init__()
        self.class_weights = torch.tensor(class_weights)
        self.threshold = threshold
        self.name = "IoU"

    @torch.no_grad()
    def forward(self, input, target):
        input = torch.argmax(input, dim = 1)
        scores = np.zeros(target.size(1))
        num_not_None = np.zeros(target.size(1))

        for i in range(1, target.shape[1]):  # background is not included
            ypr = input == i
            ygt = target[:, i, :, :]

            #scores.append(iou(ypr, ygt, threshold=self.threshold))
            true_or_flase = ygt.bool().any(dim = 1).any(dim =1) #size = (batch_size)
            num_not_None[i] += sum(true_or_flase).cpu().detach().numpy()
            scores[i] += iou(ypr[true_or_flase], ygt[true_or_flase], threshold=self.threshold, for_metric=True).cpu().detach().numpy() * sum(true_or_flase).cpu().detach().numpy()
        return scores[1:], num_not_None[1:]
