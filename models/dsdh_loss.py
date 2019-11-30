import torch.nn as nn
import torch


class DSDHLoss(nn.Module):
    def __init__(self, eta):
        super(DSDHLoss, self).__init__()
        self.eta = eta

    def forward(self, U_batch, U, S, B):
        theta = U.t() @ U_batch / 2

        # Prevent exp overflow
        theta = torch.clamp(theta, min=-100, max=50)

        metric_loss = (torch.log(1 + torch.exp(theta)) - S * theta).mean()
        quantization_loss = (B - U_batch).pow(2).mean()
        loss = metric_loss + self.eta * quantization_loss

        return loss
