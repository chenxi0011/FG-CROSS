import torch
import torch.nn.functional as F
from torch.autograd import Variable

def triplet_loss(scores, margin=0.2, max_violation=False):
    batch_size = scores.size(0)
    diagonal = scores.diag().view(batch_size, 1)

    # 加入一个小的 epsilon 防止数值不稳定
    epsilon = 1e-6
    d1 = diagonal.expand_as(scores) + epsilon
    d2 = diagonal.t().expand_as(scores) + epsilon

    # 计算图像到文本的检索损失
    cost_s = (margin + scores - d1).clamp(min=0)
    # 计算文本到图像的检索损失
    cost_im = (margin + scores - d2).clamp(min=0)

    mask = torch.eye(batch_size) > 0.5
    I = Variable(mask)
    if torch.cuda.is_available():
        I = I.cuda()
    cost_s = cost_s.masked_fill_(I, 0)
    cost_im = cost_im.masked_fill_(I, 0)

    if max_violation:
        cost_s = cost_s.max(1)[0]
        cost_im = cost_im.max(0)[0]

    return cost_s.sum() + cost_im.sum()
