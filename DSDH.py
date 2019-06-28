#!/usr/bin/env python
# -*- coding: utf-8 -*-

import models.modelloader as modelloader
import models.loss.dsdh_loss as dsdh_loss
from utils.calc_map import calc_map

import torch
import torch.optim as optim
from loguru import logger
import os
import time


def dsdh(train_dataloader,
         query_dataloader,
         train_targets,
         query_targets,
         code_length,
         max_iter,
         dcc_iter,
         mu,
         nu,
         eta,
         model,
         multi_gpu,
         device,
         lr,
         evaluate_freq,
         topk,
         ):
    """DSDH algorithm

    Parameters
        train_dataloader: DataLoader
        Training dataloader

        query_data: DataLoader
        Query dataloader

        train_targets: Tensor
        Training targets

        query_targets: Tensor
        Query targets

        code_length: int
        Hash code length

        max_iter: int
        Maximum iteration

        dcc_iter: int
        DCC iteration

        mu, nu, eta: float
        Hyper-Parameters

        model: str
        CNN model name

        multi_gpu: bool
        Is using multiple gpu

        device: str
        CPU or GPU

        lr: float
        Learning rate

        evaluate_freq: int
        Frequency of evaluating

        topk: int
        Compute mAP using top k retrieval result

    Returns
        best_model: str
        Best model name
    """
    # Construct network, optimizer, loss
    model = modelloader.load_model(model, num_classes=code_length)
    if multi_gpu:
        model = torch.nn.DataParallel(model)
    model.to(device)
    criterion = dsdh_loss.DSDHLoss(mu, nu)

    optimizer = optim.RMSprop(model.parameters(),
                              lr=lr,
                              weight_decay=10**-5,
                              )
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

    # Initialize
    N = len(train_dataloader.dataset)
    B = torch.zeros(code_length, N).to(device)
    U = torch.zeros(code_length, N).to(device)
    Y = train_targets.t()

    best_map = 0.0
    last_model = None
    total_loss = 0.0
    start = time.time()
    for itr in range(max_iter):
        # scheduler.step()
        model.train()
        for data, targets, index in train_dataloader:
            data = data.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            S = (targets @ train_targets.t() > 0).float()

            outputs = model(data)

            U[:, index] = outputs.t().data

            # DCC
            for dit in range(dcc_iter):
                # W-step
                W = torch.inverse(B @ B.t() + nu / mu * torch.eye(code_length, device=device)) @ B @ Y.t()

                # B-step
                B = solve_dcc(W, Y, U, B, eta, mu)

            loss = criterion(S, outputs, U, B, W, Y, index)
            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        if itr % evaluate_freq == evaluate_freq - 1:
            meanAP = evaluate(model,
                              query_dataloader,
                              train_dataloader,
                              query_targets,
                              train_targets,
                              code_length,
                              device,
                              topk)

            # Save best result
            if best_map < meanAP:
                if last_model:
                    os.remove(os.path.join('result', last_model))
                best_map = meanAP
                last_model = 'model_{:.4f}.t'.format(best_map)
                torch.save(model, os.path.join('result', last_model))

            logger.info('[itr: {}][time:{:.4f}][loss: {:.4f}][map: {:.4f}]'.format(itr+1,
                                                                                   time.time()-start,
                                                                                   total_loss,
                                                                                   meanAP))
            start = time.time()
            total_loss = 0.0

    return last_model


def solve_dcc(W, Y, H, B, eta, mu):
    """Solve DCC(Discrete Cyclic Coordinate Descent) problem
    """
    for i in range(B.shape[0]):
        P = W @ Y + eta / mu * H

        p = P[i, :]
        w = W[i, :]
        W_prime = torch.cat((W[:i, :], W[i+1:, :]))
        B_prime = torch.cat((B[:i, :], B[i+1:, :]))

        B[i, :] = (p - B_prime.t() @ W_prime @ w).sign()

    return B


def evaluate(model,
             query_dataloader,
             database_dataloader,
             query_targets,
             database_targets,
             code_length,
             device,
             topk,
             ):
    """Evaluate algorithm

    Parameters
        model: model
        CNN model

        query_dataloader: DataLoader
        Query dataloader

        database_dataloader: DataLoader
        Database dataloadermu,
         nu,
         eta,
         model,
         multi_gpu,
         device,
         epochs,
         lr,
         evaluate_freq,
         topk,

        query_targets: Tensor
        Query targets

        database_targets: Tensor
        Database targets

        device: str
        CPU or GPU

        topk: int
        Compute mAP using top k retrieval result

    Returns
        meanAP: float
        mean Average precision
    """
    model.eval()

    # Generate hash code
    query_code = generate_code(model, query_dataloader, code_length, device).to(device)
    database_code = generate_code(model, database_dataloader, code_length, device).to(device)

    # Compute map
    meanAP = calc_map(query_code,
                      database_code,
                      query_targets,
                      database_targets,
                      device,
                      topk,
                      )
    model.train()

    return meanAP


def generate_code(model, dataloader, code_length, device):
    """产生hash code

    Parameters
        model: Model
        CNN model

        dataloader: DataLoader
        Dataloader

        code_length: int
        Hash code length

        device: str
        GPU or CPU

    Returns
        code: Tensor
        Hash code
    """
    with torch.no_grad():
        N = len(dataloader.dataset)
        code = torch.zeros([N, code_length])
        for data, _, index in dataloader:
            data = data.to(device)
            outputs = model(data)
            code[index, :] = outputs.sign().cpu()

    return code
