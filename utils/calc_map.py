#!/usr/bin/env python
# -*- coding: utf-8 -*-

from utils.calc_hamming_dist import calc_hamming_dist

import torch


def calc_map(query_code,
             database_code,
             query_labels,
             database_labels,
             device,
             topk=None,
             ):
    """计算mAP

    Parameters
        query_code: ndarray, {-1, +1}^{m * Q}
        query的hash code

        database_code: ndarray, {-1, +1}^{n * Q}
        database的hash code

        query_labels: ndarray, {0, 1}^{m * n_classes}
        query的label，onehot编码

        database_labels: ndarray, {0, 1}^{n * n_classes}
        database的label，onehot编码

        topk: int
        计算前k个map

    Returns
        meanAP: float
        Mean Average Precision
    """
    num_query = query_labels.shape[0]
    mean_AP = 0.0

    for i in range(num_query):
        # 检索
        retrieval = (query_labels[i, :] @ database_labels.t() > 0).float()

        # hamming distance
        hamming_dist = calc_hamming_dist(query_code[i, :], database_code)

        # 根据hamming distance安排检索结果位置，并取topk个
        retrieval = retrieval[torch.argsort(hamming_dist)][:topk]

        # 检索到数量
        retrieval_cnt = retrieval.sum().int().item()

        # 未检索到
        if retrieval_cnt == 0:
            continue

        # 每个位置打分
        score = torch.linspace(1, retrieval_cnt, retrieval_cnt).to(device)

        # 检索到的下标位置
        index = (torch.nonzero(retrieval == 1).squeeze() + 1.0).float()

        mean_AP += (score / index).mean()

    mean_AP = mean_AP / num_query
    return mean_AP


