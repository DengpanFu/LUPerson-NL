#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2021-01-26 11:40:41
# @Author  : Dengpan Fu (fdpan@mail.ustc.edu.cn)

import os
import numpy as np

import torch
import torch.nn as nn

MAX_NUM = -32000

class SimCLR(nn.Module):
    """
    Build a SimCLR model with: a base encoder f() and a projection head g()
    https://arxiv.org/abs/2002.05709
    """
    def __init__(self, base_encoder, dim=128, T=0.5):
        """
        dim: feature dimension (default: 128)
        """
        super(SimCLR, self).__init__()

        self.T = T

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder = base_encoder(num_classes=dim)

        # hack: brute-force replacement
        dim_mlp = self.encoder.fc.weight.shape[1]
        self.encoder.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), 
                            nn.BatchNorm2d(dim_mlp), nn.ReLU(), self.encoder.fc)

    def forward(self, im_q=None, im_k=None):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            encoded q and k
        """
        if not self.training:
            return nn.functional.normalize(self.encoder(im_q), dim=1)

        q = self.encoder(im_q)  # queries: NxC
        q = nn.functional.normalize(q, dim=1)
        k = self.encoder(im_k)  # keys: NxC
        k = nn.functional.normalize(k, dim=1)

        all_q = concat_all_gather(q)
        all_k = concat_all_gather(k)
        all_qk = torch.cat([all_q, all_k], dim=0)

        N = all_q.shape[0]
        logits = torch.einsum('nc,ck->nk', [all_qk, all_qk.T])
        # logits[range(2*N), range(2*N)] = -10000
        index = torch.arange(2 * N, device=logits.device).repeat(2, 1)
        logits.scatter_(0, index, MAX_NUM)
        logits /= self.T


        labels_l = torch.arange(N, 2 * N)
        labels_r = torch.arange(0, N)
        labels   = torch.cat([labels_l, labels_r]).cuda()

        return logits, labels


# utils
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    # need to do this to restore propagation of the gradients
    rank = torch.distributed.get_rank()
    tensors_gather[rank] = tensor

    output = torch.cat(tensors_gather, dim=0)
    return output
