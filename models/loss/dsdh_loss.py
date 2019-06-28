# -*- coding:utf-8 -*-

import torch.nn as nn
import torch


class DSDHLoss(nn.Module):
    def __init__(self, mu, nu):
        super(DSDHLoss, self).__init__()
        self.mu = mu
        self.nu = nu

    def forward(self, S, outputs, U, B, W, Y, index):
        """
        Forward Propagation

        Parameters
            S: Tensor
            Similarity matrix

            outputs: Tensor
            CNN outputs

            U: Tensor
            Relaxation hash code

            B: Tensor
            Binary hash code

            W: Tensor
            Classification weight matrix

            Y: Tensor
            Categories targets

            index: Tensor
            Index

        Returns
            loss: Tensor
            Loss
        """
        theta = outputs @ U / 2

        # Prevent exp overflow
        theta = torch.clamp(theta, min=-100, max=50)

        loss = (torch.log(1 + torch.exp(theta)) - S * theta).sum()

        # Classification loss
        cl_loss = (Y[:, index] - W.t() @ B[:, index]).pow(2).sum()

        # Regularization loss
        reg_loss = W.pow(2).sum()

        loss = loss + self.mu * cl_loss + self.nu * reg_loss
        loss = loss / S.numel()

        return loss
