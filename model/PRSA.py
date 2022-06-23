import numpy as np
import torch
import torch.nn as nn

from model.PositionEncoding import Region_PositionEncoding
from model.RSlot_module import RSlot_module
from aligner.align import Align1DLayer

class PRSA_Net(nn.Module):
    def __init__(
            self,
            batch_size=8,
            dataset_name='thumos',
            temporal_scale=200,
            max_duration=64,
            min_duration=0,
            prop_boundary_ratio=0.5,
            num_sample=32,
            num_sample_perbin=3,
            feat_dim=2048,
            adj_snippets='4,8'
    ):
        super(PRSA_Net, self).__init__()
        self.tscale = temporal_scale
        self.max_duration = max_duration
        self.min_duration = min_duration
        self.prop_boundary_ratio = prop_boundary_ratio
        self.num_sample = num_sample
        self.num_sample_perbin = num_sample_perbin
        self.input_channel = feat_dim
        self.adj_snippets = tuple([int(t) for t in adj_snippets.split(',')])
        self.best_loss = 1e8

        self.pos = Region_PositionEncoding(self.tscale, dataset_name).pos

        self.base_unit = RSlot_module(
            in_channel=256,
            out_channel=256,
            pos=self.pos,
            t_scale=self.tscale,
            adjacency_snippets=self.adj_snippets
        )
        self.base = nn.Sequential(
            nn.Conv1d(self.input_channel, 256, kernel_size=3, padding=1),
            nn.ReLU(True),
        )

        self.start_boundary_unit = RSlot_module(
            in_channel=256,
            out_channel=256,
            pos=self.pos,
            t_scale=self.tscale,
            adjacency_snippets=self.adj_snippets
        )
        self.start = nn.Sequential(
            nn.Conv1d(256, 1, 1),
            nn.Sigmoid()
        )

        self.end_boundary_unit = RSlot_module(
            in_channel=256,
            out_channel=256,
            pos=self.pos,
            t_scale=self.tscale,
            adjacency_snippets=self.adj_snippets
        )
        self.end = nn.Sequential(
            nn.Conv1d(256, 1, 1),
            nn.Sigmoid()
        )
        self.dropout = nn.Dropout(0.5)

        self.correlation_unit = RSlot_module(
            in_channel=256,
            out_channel=256,
            pos=self.pos,
            t_scale=self.tscale,
            adjacency_snippets=self.adj_snippets
        )

        self.conf_head = nn.Sequential(
            nn.Conv2d(16 * 256, 128, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 2, kernel_size=1, padding=0),
            nn.Sigmoid()
        )
        self.roi_aligner = AlignLayer(batch_size, t_scale=self.tscale, duration=self.max_duration)

    def forward(self, x):
        base_feature, G = self.base_unit(self.base(x))
        base_feature = self.dropout(base_feature)

        start, A_s = self.start_boundary_unit(base_feature)
        start = self.start(start).squeeze(1)

        end, A_e = self.end_boundary_unit(base_feature)
        end = self.end(end).squeeze(1)
        confidence_map, A_p = self.correlation_unit(base_feature)
        confidence_map = self.roi_aligner(confidence_map)
        conf = self.conf_head(confidence_map)
        loss_conns = [A_s, A_e, A_p]
        return conf, start, end, loss_conns


class AlignLayer(nn.Module):
    def __init__(self, batch_size, t_scale=200, duration=64, expand_ratio=0.5, resolution=16):
        super(AlignLayer, self).__init__()
        self.bs = batch_size
        self.expand_ratio = expand_ratio
        self.resolution = resolution
        self.t = t_scale
        self.d = duration

        self.align_layer = Align1DLayer(self.resolution, 0)
        self._get_anchors()

    def _get_anchors(self):
        anchors = []
        for k in range(self.bs):
            for start_index in range(self.t):
                for duration_index in range(self.d):
                    if start_index + duration_index < self.t:
                        p_xmin = start_index
                        p_xmax = start_index + duration_index
                        center_len = float(p_xmax - p_xmin) + 1
                        sample_xmin = p_xmin - center_len * self.expand_ratio
                        sample_xmax = p_xmax + center_len * self.expand_ratio
                        anchors.append([k, sample_xmin, sample_xmax])
                    else:
                        anchors.append([k, 0, 0])
        self.anchor_num = len(anchors) // self.bs
        self.anchors = torch.tensor(np.stack(anchors)).float()
        return

    def forward(self, x):
        bs, ch, t = x.shape
        self.t = t
        if not self.anchors.is_cuda:
            self.anchors = self.anchors.cuda()
        anchor = self.anchors[:self.anchor_num * bs, :]

        features = self.align_layer(x, anchor)

        feat = torch.cat((features,), dim=2).view(bs, t, self.d, -1)
        return feat.permute(0, 3, 2, 1)