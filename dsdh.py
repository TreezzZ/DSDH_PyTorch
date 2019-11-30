import torch
import torch.optim as optim
import os
import time

from torch.optim.lr_scheduler import CosineAnnealingLR
from models.model_loader import load_model
from loguru import logger
from models.dsdh_loss import DSDHLoss
from utils.evaluate import mean_average_precision


def train(
    train_dataloader,
    query_dataloader,
    retrieval_dataloader,
    arch,
    code_length,
    device,
    lr,
    max_iter,
    mu,
    nu,
    eta,
    topk,
    evaluate_interval,
 ):
    """
    Training model.

    Args
        train_dataloader, query_dataloader, retrieval_dataloader(torch.utils.data.DataLoader): Data loader.
        arch(str): CNN model name.
        code_length(int): Hash code length.
        device(torch.device): GPU or CPU.
        lr(float): Learning rate.
        max_iter: int
        Maximum iteration
        mu, nu, eta(float): Hyper-parameters.
        topk(int): Compute mAP using top k retrieval result
        evaluate_interval(int): Evaluation interval.

    Returns
        checkpoint(dict): Checkpoint.
    """
    # Construct network, optimizer, loss
    model = load_model(arch, code_length).to(device)
    criterion = DSDHLoss(eta)
    optimizer = optim.RMSprop(
        model.parameters(),
        lr=lr,
        weight_decay=1e-5,
    )
    scheduler = CosineAnnealingLR(optimizer, max_iter, 1e-7)

    # Initialize
    N = len(train_dataloader.dataset)
    B = torch.randn(code_length, N).sign().to(device)
    U = torch.zeros(code_length, N).to(device)
    train_targets = train_dataloader.dataset.get_onehot_targets().to(device)
    S = (train_targets @ train_targets.t() > 0).float()
    Y = train_targets.t()
    best_map = 0.
    iter_time = time.time()

    for it in range(max_iter):
        model.train()
        # CNN-step
        for data, targets, index in train_dataloader:
            data, targets = data.to(device), targets.to(device)
            optimizer.zero_grad()

            U_batch = model(data).t()
            U[:, index] = U_batch.data
            loss = criterion(U_batch, U, S[:, index], B[:, index])

            loss.backward()
            optimizer.step()
        scheduler.step() 

        # W-step
        W = torch.inverse(B @ B.t() + nu / mu * torch.eye(code_length, device=device)) @ B @ Y.t()

        # B-step
        B = solve_dcc(W, Y, U, B, eta, mu)

        # Evaluate
        if it % evaluate_interval == evaluate_interval - 1:
            iter_time = time.time() - iter_time
            epoch_loss = calc_loss(U, S, Y, W, B, mu, nu, eta)

            # Generate hash code
            query_code = generate_code(model, query_dataloader, code_length, device)
            retrieval_code = generate_code(model, retrieval_dataloader, code_length, device)
            query_targets = query_dataloader.dataset.get_onehot_targets()
            retrieval_targets = retrieval_dataloader.dataset.get_onehot_targets()

            # Compute map
            mAP = mean_average_precision(
                query_code.to(device),
                retrieval_code.to(device),
                query_targets.to(device),
                retrieval_targets.to(device),
                device,
                topk,
            )
            logger.info('[iter:{}/{}][loss:{:.2f}][map:{:.4f}][time:{:.2f}]'.format(it+1, max_iter, epoch_loss, mAP, iter_time))

            # Save checkpoint
            if best_map < mAP:
                best_map = mAP
                checkpoint = {
                    'qB': query_code,
                    'qL': query_targets,
                    'rB': retrieval_code,
                    'rL': retrieval_targets,
                    'model': model.state_dict(),
                    'map': best_map,
                }
            iter_time = time.time()

    return checkpoint


def solve_dcc(W, Y, U, B, eta, mu):
    """
    DCC.
    """
    for i in range(B.shape[0]):
        P = W @ Y + eta / mu * U

        p = P[i, :]
        w = W[i, :]
        W_prime = torch.cat((W[:i, :], W[i+1:, :]))
        B_prime = torch.cat((B[:i, :], B[i+1:, :]))

        B[i, :] = (p - B_prime.t() @ W_prime @ w).sign()

    return B


def calc_loss(U, S, Y, W, B, mu, nu, eta):
    """
    Compute loss.
    """
    theta = torch.clamp(U.t() @ U / 2, min=-100, max=50)
    metric_loss = (torch.log(1 + torch.exp(theta)) - S * theta).sum()
    classify_loss = ((Y - W.t() @ B) ** 2).sum()
    regular_loss = (W ** 2).sum()
    quantization_loss = ((B - U) ** 2).sum()

    loss = (metric_loss + mu * classify_loss + nu * regular_loss + eta * quantization_loss) / S.shape[0]

    return loss.item()


def generate_code(model, dataloader, code_length, device):
    """
    Generate hash code

    Args
        dataloader(torch.utils.data.DataLoader): Data loader.
        code_length(int): Hash code length.
        device(torch.device): Using gpu or cpu.

    Returns
        code(torch.Tensor, n*code_length): Hash code.
    """
    model.eval()
    with torch.no_grad():
        N = len(dataloader.dataset)
        code = torch.zeros([N, code_length])
        for data, _, index in dataloader:
            data = data.to(device)
            hash_code = model(data)
            code[index, :] = hash_code.sign().cpu()

    model.train()
    return code
