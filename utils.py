# -*- coding: utf-8 -*-
import numpy as np


def ioa_with_anchors(anchor_min, anchor_max, box_min, box_max):
    if anchor_min > anchor_max:
        raise ValueError
    anchor_len = anchor_max - anchor_min
    inter_min = np.maximum(anchor_min, box_min)
    inter_max = np.minimum(anchor_max, box_max)
    len = np.maximum(inter_max - inter_min, 0.)
    if anchor_len < 1e-4:
        raise ValueError
    scores = np.divide(len, anchor_len)
    return scores


def iou_with_anchors(anchor_min, anchor_max, box_min, box_max):
    anchor_len = anchor_max - anchor_min
    inter_min = np.maximum(anchor_min, box_min)
    inter_max = np.minimum(anchor_max, box_max)
    inter_len = np.maximum(inter_max - inter_min, 0.)
    union_len = anchor_len + box_max - box_min - inter_len
    scores = np.divide(inter_len, union_len)
    return scores


def get_pos(num_node, self_link, inward, outward):
    I = edge2mat(self_link, num_node)
    In = normalize_conn(edge2mat(inward, num_node))
    outward = normalize_conn(edge2mat(outward, num_node))
    pos = I + In + outward
    return pos


def edge2mat(link, num_node):
    A = np.zeros((num_node, num_node))
    for i, j in link:
        A[i, j] = 1
    return A


def normalize_conn(A):
    Dl = np.sum(A, 0)
    h, w = A.shape
    Dn = np.zeros((w, w))
    for i in range(w):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-1)
    AD = np.dot(A, Dn)
    return AD