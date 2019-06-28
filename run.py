#!/usr/bin/env python
# -*- coding: utf-8 -*-

import DSDH
import data.dataloader as dataloader
from data.transform import encode_onehot
from DSDH import evaluate

import argparse
import torch
import os
from loguru import logger


def run_dsdh(opt):
    """Run DSDH algorithm

    Parameters
        opt: parser
        Configuration

    Returns
        None
    """
    # Load data
    query_dataloader, train_dataloader, database_dataloader = dataloader.load_data(opt)

    # onehot targets
    if opt.dataset == 'cifar10':
        query_targets = torch.FloatTensor(encode_onehot(query_dataloader.dataset.targets)).to(opt.device)
        train_targets = torch.FloatTensor(encode_onehot(train_dataloader.dataset.targets)).to(opt.device)
        database_targets = torch.FloatTensor(encode_onehot(database_dataloader.dataset.targets)).to(opt.device)
    elif opt.dataset == 'nus-wide':
        query_targets = torch.FloatTensor(query_dataloader.dataset.targets).to(opt.device)
        train_targets = torch.FloatTensor(train_dataloader.dataset.targets).to(opt.device)
        database_targets = torch.FloatTensor(database_dataloader.dataset.targets).to(opt.device)

    cl = [12, 24, 32, 48]
    for c in cl:
        opt.code_length = c

        # DSDH algorithm
        logger.info(opt)
        best_model = DSDH.dsdh(train_dataloader,
                               query_dataloader,
                               train_targets,
                               query_targets,
                               opt.code_length,
                               opt.max_iter,
                               opt.dcc_iter,
                               opt.mu,
                               opt.nu,
                               opt.eta,
                               opt.model,
                               opt.multi_gpu,
                               opt.device,
                               opt.lr,
                               opt.evaluate_freq,
                               opt.topk,
                               )

        # Evaluate whole dataset
        model = torch.load(os.path.join('result', best_model))
        final_map = evaluate(model,
                             query_dataloader,
                             database_dataloader,
                             query_targets,
                             database_targets,
                             opt.code_length,
                             opt.device,
                             opt.topk,
                             )
        logger.info('final_map: {:.4f}'.format(final_map))


def load_parse():
    """Load configuration

    Parameters
        None

    Returns
        opt: parser
        Configuration
    """
    parser = argparse.ArgumentParser(description='DSDH_PyTorch')
    parser.add_argument('--dataset', default='cifar10', type=str,
                        help='Dataset used to train (default: cifar10)')
    parser.add_argument('--data-path', type=str,
                        help='Path of cifar10 dataset')
    parser.add_argument('--num-query', default=1000, type=int,
                        help='Number of query(default: 1000)')
    parser.add_argument('--num-train', default=5000, type=int,
                        help='Number of train(default: 5000)')
    parser.add_argument('--code-length', default=12, type=int,
                        help='Binary hash code length (default: 12)')
    parser.add_argument('--topk', default=5000, type=int,
                        help='Compute map of top k (default: 5000)')
    parser.add_argument('--evaluate-freq', default=10, type=int,
                        help='Frequency of evaluate (default: 10)')
    parser.add_argument('--max-iter', default=150, type=int,
                        help='Maximum iteration (default: 150)')
    parser.add_argument('--dcc-iter', default=10, type=int,
                        help='Dcc iteration with one epoch')

    parser.add_argument('--model', default='alexnet', type=str,
                        help='CNN model(default: alexnet)')
    parser.add_argument('--multi-gpu', action='store_true',
                        help='Use multiple gpu')
    parser.add_argument('--gpu', default=0, type=int,
                        help='Use gpu(default: 0. -1: use cpu)')
    parser.add_argument('--lr', default=1e-5, type=float,
                        help='Learning rate(default: 1e-5)')
    parser.add_argument('--batch-size', default=256, type=int,
                        help='Batch size(default: 256)')
    parser.add_argument('--num-workers', default=0, type=int,
                        help='Number of workers(default: 0)')

    parser.add_argument('--nu', default=0.1, type=float,
                        help='Hyper-parameter (default: 0.1)')
    parser.add_argument('--mu', default=1, type=float,
                        help='Hyper-parameter (default: 1)')
    parser.add_argument('--eta', default=55, type=int,
                        help='Hyper-parameter (default: 55)')

    return parser.parse_args()


def set_seed(seed):
    import random
    import torch.backends.cudnn as cudnn
    random.seed(seed)
    torch.manual_seed(seed)
    cudnn.deterministic = True


if __name__ == "__main__":
    opt = load_parse()
    logger.add('logs/file_{time}.log')

    # set_seed(20180707)

    if opt.gpu == -1:
        opt.device = torch.device("cpu")
    else:
        opt.device = torch.device("cuda:%d" % opt.gpu)

    run_dsdh(opt)
