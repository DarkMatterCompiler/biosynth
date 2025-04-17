import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class LargeMarginSoftmax(nn.CrossEntropyLoss):
    def __init__(self, reg_lambda=0.3, weight=None, ignore_index=-100, reduction='mean'):
        super().__init__(weight=weight, ignore_index=ignore_index, reduction=reduction)
        self.reg_lambda = reg_lambda

    def forward(self, inputs, target):
        N, C = inputs.size()
        mask = torch.zeros_like(inputs)
        mask[range(N), target] = 1

        ce_loss = F.cross_entropy(inputs, target, reduction=self.reduction)

        X = inputs - 1e6 * mask
        softmax_X = F.softmax(X, dim=1).clamp(min=1e-6, max=1.0)
        log_softmax_X = F.log_softmax(X, dim=1).clamp(min=-100, max=0)

        reg = 0.5 * ((softmax_X - 1.0 / (C - 1)) *
                     log_softmax_X * (1.0 - mask)).sum(dim=1)
        reg = reg.mean() if self.reduction == 'mean' else reg.sum()

        return ce_loss + self.reg_lambda * reg


class ContrastiveLossModule(nn.Module):
    def __init__(self, alpha=0.8, theta=0.04, temperature=1.0, device='cuda'):
        super().__init__()
        self.alpha = alpha
        self.theta = theta
        self.device = device

    def forward(self, emb_x, emb_y, labels):
        # Normalize with epsilon for stability
        emb_x = F.normalize(emb_x + 1e-6, dim=1)
        emb_y = F.normalize(emb_y + 1e-6, dim=1)

        sim_matrix = torch.exp((emb_x @ emb_y.T) / self.theta)
        sim_matrix = sim_matrix.clamp(min=1e-6, max=1e6)

        batch_size = emb_x.size(0)
        same_id = labels.unsqueeze(1) == labels.unsqueeze(0)
        diff_id = ~same_id
        eye = torch.eye(batch_size, dtype=torch.bool, device=labels.device)
        same_id = same_id ^ eye  # remove diagonal

        sigma_xy = sim_matrix[torch.arange(
            batch_size), torch.arange(batch_size)]

        numerator = sigma_xy
        denom = (
            sigma_xy +
            self.alpha * sim_matrix[same_id].view(batch_size, -1).sum(dim=1) +
            sim_matrix[diff_id].view(batch_size, -1).sum(dim=1)
        ).clamp(min=1e-6)

        # Safe log
        loss = -torch.log((numerator / denom).clamp(min=1e-6))
        return loss.mean()


class TotalLoss(nn.Module):
    def __init__(self, num_classes, device='cuda'):
        super().__init__()
        self.lm_f = LargeMarginSoftmax()
        self.lm_p = LargeMarginSoftmax()
        self.cl_fp = ContrastiveLossModule(device=device)
        self.cl_fa = ContrastiveLossModule(device=device)
        self.cl_pa = ContrastiveLossModule(device=device)

    def forward(self, logits_f, logits_p, emb_f, emb_p, emb_a, labels):
        l_lm_f = self.lm_f(logits_f, labels)
        l_lm_p = self.lm_p(logits_p, labels)

        l_cl = self.cl_fp(emb_f, emb_p, labels) + \
            self.cl_fa(emb_f, emb_a, labels) + \
            self.cl_pa(emb_p, emb_a, labels)

        return l_lm_f + l_lm_p + l_cl
