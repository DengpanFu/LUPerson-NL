#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2021-03-09 16:52:56
# @Author  : Dengpan Fu (fdpan@mail.ustc.edu.cn)

import os
import numpy as np

import torch
import torch.nn as nn
from random import sample
import numpy as np
import torch.nn.functional as F

class LNL(nn.Module):
    def __init__(self, base_encoder, backbone, dim=128, K=65536, m1=0.999, m2=0.999, 
        T=0.1, cls_num=1000, alpha=0.5, pseudo_th=0.8, cls_dim=None):
        super(LNL, self).__init__()
        self.dim       = dim
        self.K         = K    # queue size
        self.m1        = m1   # momentum for encoder
        self.m2        = m2   # momentum for prototypes
        self.T         = T    # temperature
        self.cls_num   = cls_num
        self.alpha     = alpha
        self.pseudo_th = pseudo_th
        #encoder
        self.encoder_q = base_encoder(backbone, embed_dim=dim, cls_num=cls_num, cls_dim=cls_dim)
        #momentum encoder
        self.encoder_k = base_encoder(backbone, embed_dim=dim, cls_num=cls_num, cls_dim=cls_dim)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.register_buffer("queue_id", -torch.ones((K, 1), dtype=torch.long))
        self.queue = F.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("prototypes", torch.zeros(cls_num, dim))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        update momentum encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m1 + param_q.data * (1. - self.m1)


    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, ids):
        # gather keys before updating queue
        keys = concat_all_gather(keys)
        ids = concat_all_gather(ids)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)

        if ptr + batch_size <= self.K:
            self.queue[:, ptr:ptr + batch_size] = keys.T
            self.queue_id[ptr:ptr + batch_size, 0] = ids
            ptr = ptr + batch_size
        else:
            remain = ptr + batch_size - self.K
            self.queue[:, ptr:] = keys[:self.K - ptr].T
            self.queue_id[ptr:, 0] = ids[:self.K - ptr]
            
            self.queue[:, :remain] = keys[self.K - ptr:].T
            self.queue_id[:remain, 0] = ids[self.K - ptr:]
            ptr = remain

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]
    

    def forward(self, im_q=None, im_k=None, target=None, is_proto=False, 
        is_clean=False, is_supcont=False, topk=()):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
            target: initial class label for input images (may have noise)
            is_proto: if compute prototypical contrastive loss
            is_clean: if clean labels
        Output:
            output: classification prediction
            target: refined class labels
            logits: momentum feature logits prediction
            inst_labels: instance labels
            logits_proto: logits computed with prototype
        """
        if not self.training:
            return self.encoder_q(im_q)
        else:
            assert(not target is None), 'target must provided'

        output, q = self.encoder_q(im_q)

        # compute augmented features 
        with torch.no_grad():  # no gradient 
            self._momentum_update_key_encoder()  # update the momentum encoder
            # shuffle for making use of BN
            im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)
            _, k = self.encoder_k(im_k)
            # undo shuffle
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        # compute instance logits
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
        logits = torch.cat([l_pos, l_neg], dim=1)
        # apply temperature
        logits /= self.T
        
        # constrast loss
        if not is_supcont:
            mask = torch.zeros_like(logits)
            mask[:, 0] = 1.
        else:
            queue_ids = self.queue_id.clone().detach()
            input_ids = target.clone().detach().unsqueeze(1)
            mask = torch.eq(input_ids, queue_ids.T).float()
            aug_pos = torch.ones_like(input_ids)
            mask = torch.cat([aug_pos, mask], dim=1)

        inst_loss = torch.mean(-(torch.log_softmax(logits, dim=1) * 
                               mask).sum(dim=1) / mask.sum(1))

        if len(topk):
            inst_accs = logits_accuracy(logits, mask, topk)
        else:
            inst_accs = None

        if is_proto:
            # compute protoypical logits
            prototypes = self.prototypes.clone().detach()
            logits_proto = torch.mm(q, prototypes.t()) / self.T
        else:
            logits_proto = 0

        if is_clean:
            # noise cleaning
            soft_label = self.alpha * F.softmax(output, dim=1) + \
                         (1 - self.alpha) * F.softmax(logits_proto, dim=1)
            # assign a new pseudo label
            max_score, hard_label = soft_label.max(dim=1)
            correct_idx = max_score > self.pseudo_th
            target[correct_idx] = hard_label[correct_idx]
        
        # dequeue and enqueue
        self._dequeue_and_enqueue(k, target)

        # aggregate features and (pseudo) labels across all gpus
        targets  = concat_all_gather(target)
        features = concat_all_gather(q)

        # update momentum prototypes with original labels
        for feat, label in zip(features, targets):
            self.prototypes[label] = self.prototypes[label] * self.m2 + (1 - self.m2) * feat

        # normalize prototypes
        self.prototypes = F.normalize(self.prototypes, p=2, dim=1)

        return output, target, inst_loss, logits_proto, inst_accs


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output

@torch.no_grad()
def logits_accuracy(logits, masks, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = logits.size(0)

        _, preds = logits.topk(maxk, 1, True, True)
        corrects = masks.gather(dim=1, index=preds)

        res = []
        for k in topk:
            correct_k = (corrects[:, :k].sum(1) >= 1).float().sum()
            res.append(correct_k.mul_(100.0 / batch_size))
        return res