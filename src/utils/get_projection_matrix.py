#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    get_projection_matrix.py
# @Time:        22/03/2023 12.58

import numpy as np
import torch


def whitening(embs):
    embs = torch.from_numpy(embs)
    mu = torch.mean(embs, dim=0, keepdim=True)
    cov = torch.cov(embs.T)
    U, s, V = torch.linalg.svd(cov)
    W = torch.mm(U, torch.diag(1 / torch.sqrt(s)))
    return W, -mu


def transform_and_normalize(embs, kernel, bias):
    embs = torch.from_numpy(embs) if isinstance(embs, np.ndarray) else embs
    if not (kernel is None or bias is None):
        embs = torch.mm(embs + bias, kernel)
    normalized_embs = embs / torch.norm(embs, dim=1, keepdim=True)

    return normalized_embs
