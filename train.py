import time

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from model.PRSA import PRSA_Net
# from model.loss_function import get_mask
from model.loss_function import Criterion
from dataset import VideoDataset
from engine import train_epoch, test


def Train(args):
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
    criterion = Criterion(tscale=args.dataset['temporal_scale'], duration=args.dataset['max_duration'])

    model = torch.nn.DataParallel(model, ).cuda()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                           lr=args.scheme['training_lr'], weight_decay=args.scheme['weight_decay'])

    train_loader = DataLoader(
        VideoDataset(
            temporal_scale=args.dataset['temporal_scale'],
            mode=args.mode,
            subset="train",
            feature_path=args.dataset['feature_path'],
            video_info_path=args.dataset['video_info_path'],
            feat_dim=args.model['feat_dim'],
            gap_videoframes=args.dataset['gap_videoframes'],
            max_duration=args.dataset['max_duration'],
            min_duration=args.dataset['max_duration'],
            feature_name=args.dataset['feature_name'],
            overwrite=args.dataset['overwrite']
        ),
        batch_size=args.scheme['batch_size'],
        shuffle=True,
        num_workers=args.dataset['num_workers'],
        pin_memory=True
    )
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
        batch_size=args.scheme['batch_size'],
        shuffle=False,
        num_workers=args.dataset['num_workers'],
        pin_memory=True
    )

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.scheme['step_size'], gamma=args.scheme['step_gamma'])

    # bm_mask = get_mask(args.dataset['temporal_scale'], args.dataset['max_duration'])

    for epoch in range(args.scheme['train_epoch']):
        print("epoch start time:%s" % time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        train_epoch(train_loader, model, criterion, optimizer, epoch)
        test(args.output, test_loader, model, criterion, epoch)
        scheduler.step()