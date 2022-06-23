# -*- coding: utf-8 -*-
from utils import ioa_with_anchors, iou_with_anchors

import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
import os
import h5py
import pickle


class VideoDataset(data.Dataset):
    def __init__(
            self,
            temporal_scale=200,
            mode='train',
            subset="train",
            feature_path='',
            video_info_path='',
            feat_dim=2048,
            gap_videoframes=4,
            max_duration=64,
            min_duration=0,
            feature_name='i3d',
            overwrite=False
    ):
        self.temporal_scale = temporal_scale
        self.temporal_gap = 1. / self.temporal_scale
        self.subset = subset
        self.mode = mode
        self.feature_path = feature_path
        self.video_info_path = video_info_path
        self.feature_dim = feat_dim

        self.gap_videoframes = gap_videoframes

        self.num_videoframes = self.temporal_scale
        self.max_duration = max_duration
        self.min_duration = min_duration
        self.feat_name = feature_name
        self.overwrite = overwrite

        self._get_data()
        self.video_list = self.data['video_names']
        self._get_match_map()

    def _get_video_data(self, data, index):
        return data['video_data'][index]

    def _get_match_map(self):
        match_map = []
        for idx in range(self.num_videoframes):
            tmp_match_window = []
            xmin = self.temporal_gap * idx
            for jdx in range(1, self.max_duration + 1):
                xmax = xmin + self.temporal_gap * jdx
                tmp_match_window.append([xmin, xmax])
            match_map.append(tmp_match_window)
        match_map = np.array(match_map)
        match_map = match_map.transpose([1, 0, 2])
        match_map = match_map.reshape([-1, 2])
        self.match_map = match_map
        self.anchor_xmin = [self.temporal_gap * (i - 0.5) for i in range(self.temporal_scale)]
        self.anchor_xmax = [self.temporal_gap * (i + 0.5) for i in range(1, self.temporal_scale + 1)]

    def _get_data(self):
        global anno_df
        if self.subset == 'train':
            anno_df = pd.read_csv(self.video_info_path + 'val_Annotation.csv')

        elif self.subset == 'val':
            anno_df = pd.read_csv(self.video_info_path + 'test_Annotation.csv')
        video_name_list = sorted(list(set(anno_df.video.values[:])))

        data_path = os.path.join('./data', 'saved.%s.%s.nf%d.sf%d.num%d.%s.pkl' % (
            self.feature_dim, self.subset, self.num_videoframes, self.gap_videoframes,
            len(video_name_list), self.mode))
        print('will dump data in ', data_path)

        if os.path.exists(data_path) and not self.overwrite:
            with open(data_path, 'rb') as f:
                self.data, self.durations = pickle.load(f)
            print(f'get saved data from: {data_path}')
            return

        num_videoframes = self.num_videoframes
        gap_videoframes = self.gap_videoframes

        stride = int(num_videoframes / 2)

        list_data = []
        list_anchor_xmins = []
        list_anchor_xmaxs = []
        list_gt_bbox = []
        list_videos = []
        list_indices = []

        self.duration = {}

        if self.feat_name == 'tsn':
            self.flow_val = h5py.File(self.feature_path + '/flow_val.h5', 'r')
            self.rgb_val = h5py.File(self.feature_path + '/rgb_val.h5', 'r')
            self.flow_test = h5py.File(self.feature_path + '/flow_test.h5', 'r')
            self.rgb_test = h5py.File(self.feature_path + '/rgb_test.h5', 'r')

        for num_video, video_name in enumerate(video_name_list):
            anno_df_video = anno_df[anno_df.video == video_name]
            if self.mode == 'train':
                gt_xmins = anno_df_video.startFrame.values[:]
                gt_xmaxs = anno_df_video.endFrame.values[:]
            if self.feat_name == 'tsn':
                if 'val' in video_name:
                    features = [
                        self.flow_val[video_name][::self.gap_videoframes, ...],
                        self.rgb_val[video_name][::self.gap_videoframes, ...]
                    ]
                elif 'test' in video_name:
                    features = [
                        self.flow_test[video_name][::self.gap_videoframes, ...],
                        self.rgb_test[video_name][::self.gap_videoframes, ...]
                    ]

                num_snippet = min([feature.shape[0] for feature in features])
                df_data = np.concatenate([feature[:num_snippet, :] for feature in features], axis=1)

            elif self.feat_name == 'i3d':
                features = np.load(os.path.join(self.feature_path, video_name + '.npy'))
                num_snippet = features.shape[0]
                df_data = features
            else:
                raise FileNotFoundError

            df_snippet = [gap_videoframes * i for i in range(num_snippet)]
            num_windows = int((num_snippet + stride - num_videoframes) / stride)
            window_start = [i * stride for i in range(num_windows)]

            if num_snippet < num_videoframes:
                window_start = [0]
                tmp_data = np.zeros((num_videoframes - num_snippet, self.feature_dim))
                df_data = np.concatenate((df_data, tmp_data), axis=0)

                df_snippet.extend([
                    df_snippet[-1] + gap_videoframes * (i + 1) for i in range(num_videoframes - num_snippet)
                ])
            elif num_snippet - window_start[-1] - num_videoframes > int(num_videoframes / gap_videoframes):
                window_start.append(num_snippet - num_videoframes)

            for start in window_start:
                tmp_data = df_data[start:start + num_videoframes, :]
                tmp_snippets = np.array(df_snippet[start:start + num_videoframes])

                if self.mode == 'train':
                    tmp_anchor_xmins = tmp_snippets - gap_videoframes / 2.
                    tmp_anchor_xmaxs = tmp_snippets + gap_videoframes / 2.
                    tmp_gt_bbox = []
                    tmp_ioa_list = []
                    for idx in range(len(gt_xmins)):
                        tmp_ioa = ioa_with_anchors(gt_xmins[idx], gt_xmaxs[idx],
                                                   tmp_anchor_xmins[0], tmp_anchor_xmaxs[-1])
                        tmp_ioa_list.append(tmp_ioa)
                        if tmp_ioa > 0:
                            tmp_gt_bbox.append([gt_xmins[idx], gt_xmaxs[idx]])

                    if len(tmp_gt_bbox) > 0 and max(tmp_ioa_list) > 0.9:
                        list_gt_bbox.append(tmp_gt_bbox)
                        list_anchor_xmins.append(tmp_anchor_xmins)
                        list_anchor_xmaxs.append(tmp_anchor_xmaxs)
                        list_videos.append(video_name)
                        list_indices.append(tmp_snippets)
                        if self.feature_path:
                            list_data.append(np.array(tmp_data).astype(np.float32))
                elif "infer" in self.mode:
                    list_videos.append(video_name)
                    list_indices.append(tmp_snippets)
                    list_data.append(np.array(tmp_data).astype(np.float32))

        print("load videos: ", len(set(list_videos)), flush=True)
        self.data = {
            'video_names': list_videos,
            'indices': list_indices
        }
        if self.mode == 'train':
            self.data.update({
                'gt_bbox': list_gt_bbox,
                'anchor_xmins': list_anchor_xmins,
                'anchor_xmaxs': list_anchor_xmaxs

            })
        if self.feature_path:
            self.data['video_data'] = list_data

        print('Size of data:', len(self.data['video_names']), flush=True)

        print('Dumped data...')
        with open(data_path, 'wb') as f:
            pickle.dump([self.data, self.duration], f)

    def __getitem__(self, index):
        video_data = self._get_video_data(self.data, index)
        video_data = torch.tensor(video_data.transpose())
        if self.mode == 'train':
            match_score_start, match_score_end, confidence_score = self._get_train_label(index)
            return video_data, confidence_score, match_score_start, match_score_end
        else:
            return index, video_data

    def _get_train_label(self, index):
        gt_iou_map = []
        gt_bbox = self.data['gt_bbox'][index]
        anchor_xmin = self.data['anchor_xmins'][index]
        anchor_xmax = self.data['anchor_xmaxs'][index]

        offset = int(min(anchor_xmin))

        for j in range(len(gt_bbox)):
            tmp_start = max(min(1, (gt_bbox[j][0] - offset) * self.temporal_gap / self.gap_videoframes), 0)
            tmp_end = max(min(1, (gt_bbox[j][1] - offset) * self.temporal_gap / self.gap_videoframes), 0)

            tmp_gt_iou_map = iou_with_anchors(
                self.match_map[:, 0], self.match_map[:, 1], tmp_start, tmp_end
            )
            tmp_gt_iou_map = tmp_gt_iou_map.reshape(self.max_duration, self.num_videoframes)
            gt_iou_map.append(tmp_gt_iou_map)
        gt_iou_map = np.array(gt_iou_map)
        gt_iou_map = np.max(gt_iou_map, axis=0)
        gt_iou_map = torch.Tensor(gt_iou_map)

        gt_bbox = np.array(gt_bbox)
        gt_xmins = gt_bbox[:, 0]
        gt_xmaxs = gt_bbox[:, 1]

        gt_len_small = 3 * self.gap_videoframes

        gt_start_bboxs = np.stack((gt_xmins - gt_len_small / 2, gt_xmins + gt_len_small / 2), axis=1)
        gt_end_bboxs = np.stack((gt_xmaxs - gt_len_small / 2, gt_xmaxs + gt_len_small / 2), axis=1)


        match_score_start = []
        for jdx in range(len(anchor_xmin)):
            match_score_start.append(np.max(
                ioa_with_anchors(anchor_xmin[jdx], anchor_xmax[jdx], gt_start_bboxs[:, 0], gt_start_bboxs[:, 1])
            ))
        match_score_end = []
        for idx in range(len(anchor_xmin)):
            match_score_end.append(np.max(
                ioa_with_anchors(anchor_xmin[idx], anchor_xmax[idx], gt_end_bboxs[:, 0], gt_end_bboxs[:, 1])
            ))

        match_score_start = torch.Tensor(match_score_start)
        match_score_end = torch.Tensor(match_score_end)

        return match_score_start, match_score_end, gt_iou_map

    def __len__(self):
        return len(self.video_list)