#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2021-01-31 20:36:12
# @Author  : Dengpan Fu (fdpan@mail.ustc.edu.cn)

import os
import numpy as np

import torch
import torch.nn as nn


class SupCont(nn.Module):
    def __init__(self, base_encoder, dim=128, mlp=True, has_bn=True):
        """
        dim: feature dimension (default: 128)
        """
        super(SupCont, self).__init__()

        # create the encoders
        self.encoder = base_encoder(num_classes=dim)
        self.mlp = mlp

        if mlp:
            dim_mlp = self.encoder.fc.weight.shape[1]
            if has_bn:
                self.encoder.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), 
                        nn.BatchNorm2d(dim_mlp), nn.ReLU(), self.encoder.fc)
            else:
                self.encoder.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), 
                                                 nn.ReLU(), self.encoder.fc)

    def forward(self, im_q=None, im_k=None, label=None):
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
        all_qk = torch.cat([all_q.unsqueeze(1), all_k.unsqueeze(1)], dim=1)
        all_label = concat_all_gather(label)

        return all_qk, all_label


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
