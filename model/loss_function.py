# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Criterion(nn.Module):
    def __init__(self, tscale, duration):
        super(Criterion, self).__init__()
        self.prop_mask = self.get_mask(tscale, duration)

    def get_mask(self, tscale, duration):
        mask = []
        for idx in range(duration):
            mask_vector = [1 for i in range(tscale - idx)] + [0 for i in range(idx)]
            mask.append(mask_vector)

        return torch.tensor(np.array(mask, dtype=np.float32), requires_grad=False)

    def sparsity_loss(self, loss_G):
        def per_loss(G):
            l1 = torch.mean(torch.abs(G))
            return l1

        loss_s = per_loss(loss_G[0])
        loss_e = per_loss(loss_G[1])
        loss_p = per_loss(loss_G[2])
        loss = loss_s + loss_e + loss_p
        return loss

    def binary_logistic_regression(self, pre_start, pred_end, gt_start, gt_end):
        def bi_loss(pred_score, gt_label):
            pred_score = pred_score.view(-1)
            gt_label = gt_label.view(-1)
            pmask = (gt_label > 0.5).float()

            num_entries = len(pmask)
            num_positive = torch.sum(pmask)
            ratio = num_entries / num_positive
            coef_0 = 0.5 * ratio / (ratio - 1)
            coef_1 = 0.5 * ratio
            epsilon = 0.000001
            loss_pos = coef_1 * torch.log(pred_score + epsilon) * pmask
            loss_neg = coef_0 * torch.log(1.0 - pred_score + epsilon) * (1.0 - pmask)
            loss = -1 * torch.mean(loss_pos + loss_neg)
            return loss

        loss_start = bi_loss(pre_start, gt_start)
        loss_end = bi_loss(pred_end, gt_end)
        loss = loss_end + loss_start
        return loss

    def proposal_mse_loss(self, pred_score, gt_iou_map, mask):
        u_hmask = (gt_iou_map > 0.7).float()
        u_mmask = ((gt_iou_map <= 0.7) & (gt_iou_map > 0.3)).float()
        u_lmask = ((gt_iou_map <= 0.3) & (gt_iou_map > 0.)).float()
        u_lmask = u_lmask * mask

        num_h = torch.sum(u_hmask)
        num_m = torch.sum(u_mmask)
        num_l = torch.sum(u_lmask)

        r_m = num_h / num_m
        u_smmask = torch.Tensor(np.random.rand(*gt_iou_map.shape)).cuda()
        u_smmask = u_mmask * u_smmask

        u_smmask = (u_smmask > (1. - r_m)).float()

        r_l = num_h / num_l
        u_slmask = torch.Tensor(np.random.rand(*gt_iou_map.shape)).cuda()
        u_slmask = u_lmask * u_slmask

        u_slmask = (u_slmask > (1. - r_l)).float()

        weights = u_hmask + u_smmask + u_slmask

        loss = F.mse_loss(pred_score * weights, gt_iou_map * weights)
        loss = 0.5 * torch.sum(loss * torch.ones(*weights.shape).cuda()) / torch.sum(weights)

        return loss

    def proposal_blr_loss(self, pred_score, gt_iou_map, mask):
        pmask = (gt_iou_map > 0.9).float()

        nmask = (gt_iou_map <= 0.9).float()
        nmask = nmask * mask

        num_positive = torch.sum(pmask)
        num_entries = num_positive + torch.sum(nmask)
        ratio = num_entries / num_positive
        coef_0 = 0.5 * ratio / (ratio - 1)
        coef_1 = 0.5 * ratio
        epsilon = 0.000001
        loss_pos = coef_1 * torch.log(pred_score + epsilon) * pmask
        loss_neg = coef_0 * torch.log(1.0 - pred_score + epsilon) * nmask
        loss = -1 * torch.sum(loss_pos + loss_neg) / num_entries
        return loss


    def forward(self, conf, p_start, p_end, conns, gt_iou_map, gt_start, gt_end):
        pro_mask = self.prop_mask.cuda(conf.device)

        conf_com = conf[:, 0].contiguous()
        conf_cls = conf[:, 1].contiguous()

        gt_iou_map = gt_iou_map * pro_mask

        loss_b = self.binary_logistic_regression(p_start, p_end, gt_start, gt_end)
        loss_com = self.proposal_mse_loss(conf_com, gt_iou_map, pro_mask)
        loss_cls = self.proposal_blr_loss(conf_cls, gt_iou_map, pro_mask)
        loss_norm = self.sparsity_loss(conns)

        loss = loss_b + 100 * loss_com + loss_cls + 10 * loss_norm

        return loss, loss_b, loss_com, loss_cls, loss_norm
