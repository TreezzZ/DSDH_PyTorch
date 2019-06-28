#!/usr/bin/env python
# -*- coding: utf-8 -*-


def calc_hamming_dist(B1, B2):
    """计算B1，B2间的hamming distance

    Parameters
        B1, B2: Tensor
        hash code

    Returns
        dist: Tensor
        hamming distance
    """
    return 0.5 * (B2.shape[1] - B1 @ B2.t())
