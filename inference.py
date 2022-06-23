import os
import math

import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from model.PRSA import PRSA_Net
from dataset import VideoDataset
from Evaluation.eval import evaluation_proposal,evalution_detection
import Evaluation.post_processing_d as eval_d
import Evaluation.post_processing_p as eval_p


def inference(args):
    model = PRSA_Net(
        batch_size=args.scheme['batch_size'],
        dataset_name=args.dataset['dataset_name'],
        temporal_scale=args.dataset['temporal_scale'],
        max_duration=args.dataset['max_duration'],
        min_duration=args.dataset['min_duration'],
        prop_boundary_ratio=args.model['prop_boundary_ratio'],
        num_sample=args.model['num_sample'],
        num_sample_perbin=args.model['num_sample_perbin'],
        feat_dim=args.model['feat_dim']
    )

    model = torch.nn.DataParallel(model, ).cuda()
    checkpoint = torch.load(os.path.join(args.output["checkpoint_path"], args.eval['eval_model']))
    # checkpoint = torch.load(os.path.join(args.output["checkpoint_path"], "PRSA_checkpoint_4.pth.tar"))
    print("load epoch :", checkpoint['epoch'])
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    test_loader = DataLoader(
        VideoDataset(
            temporal_scale=args.dataset['temporal_scale'],
            mode=args.mode,
            subset="val",
            feature_path=args.dataset['feature_path'],
            video_info_path=args.dataset['video_info_path'],
            feat_dim=args.model['feat_dim'],
            gap_videoframes=args.dataset['gap_videoframes'],
            max_duration=args.dataset['max_duration'],
            min_duration=args.dataset['max_duration'],
            feature_name=args.dataset['feature_name'],
            overwrite=args.dataset['overwrite']
        ),
        batch_size=1,
        shuffle=False,
        num_workers=args.dataset['num_workers'],
        pin_memory=True,
        drop_last=False
    )
    tscale = args.dataset['temporal_scale']
    duration = args.dataset['max_duration']
    with torch.no_grad():
        for idx, input_data in test_loader:
            video_name = test_loader.dataset.video_list[idx[0]]
            offset = min(test_loader.dataset.data['indices'][idx[0]])

            video_name = video_name + '_{}'.format(math.floor(offset / 250))

            input_data = input_data.cuda()
            # 1,2,D,T   1,1,T   1,1,T
            confidence_map, start, end, loss_G = model(input_data)

            start_scores = start[0].detach().cpu().numpy()
            end_scores = end[0].detach().cpu().numpy()
            clr_confidence = (confidence_map[0][1]).detach().cpu().numpy()
            reg_confidence = (confidence_map[0][0]).detach().cpu().numpy()

            max_start = max(start_scores)
            max_end = max(end_scores)

            start_bins = np.zeros(len(start_scores))
            start_bins[0] = 1
            for idx in range(1, tscale - 1):
                if start_scores[idx] > start_scores[idx + 1] and start_scores[idx] > start_scores[idx - 1]:
                    start_bins[idx] = 1
                elif start_scores[idx] > (0.5 * max_start):
                    start_bins[idx] = 1
            end_bins = np.zeros(len(end_scores))
            end_bins[-1] = 1
            for idx in range(1, tscale - 1):
                if end_scores[idx] > end_scores[idx + 1] and end_scores[idx] > end_scores[idx - 1]:
                    end_bins[idx] = 1
                elif end_scores[idx] > (0.5 * max_end):
                    end_bins[idx] = 1

            # generate proposal
            ################### THUMOS dataset########
            new_props = []
            for idx in range(duration):
                for jdx in range(tscale):
                    start_index = jdx
                    end_index = start_index + idx + 1
                    if end_index < tscale and start_bins[start_index] == 1 and end_bins[end_index] == 1:
                        xmin = start_index * args.dataset['gap_videoframes'] + offset
                        #
                        xmax = end_index * args.dataset['gap_videoframes'] + offset
                        xmin_score = start_scores[start_index]
                        xmax_score = end_scores[end_index]
                        clr_score = clr_confidence[idx, jdx]
                        reg_score = reg_confidence[idx, jdx]
                        score = xmin_score * xmax_score * clr_score * reg_score

                        new_props.append([xmin, xmax, xmin_score, xmax_score, clr_score, reg_score, score])

            new_props = np.stack(new_props)

            col_name = ['xmin', 'xmax', 'xmin_score', 'xmax_score', 'clr_score', 'reg_score', 'score']
            new_df = pd.DataFrame(new_props, columns=col_name)
            if not os.path.exists(args.output['output_path']):
                os.makedirs(args.output['output_path'])
            new_df.to_csv(os.path.join(args.output['output_path'], video_name + ".csv"), index=False)
        print("end")

def eval_proposal(args):
    args.eval['NMS_threshold'] = args.eval['NMS_threshold_p']
    eval_p.gen_proposal_muticore(args)
    evaluation_proposal(args)

def eval_detection(args):
    args.eval['NMS_threshold'] = args.eval['NMS_threshold_d']
    eval_d.gen_proposal_muticore(args)
    evalution_detection(args)







