import os

import torch

def train_epoch(data_loader, model, criterion, optimizer, epoch):
    model.train()
    epoch_prop_com_loss = 0
    epoch_prop_cls_loss = 0
    epoch_b_loss = 0
    epoch_loss = 0
    epoch_norm_loss = 0
    loss_list = []

    for iter, (input_data, label_confidence, label_start, label_end) in enumerate(data_loader):

        input_data = input_data.cuda()
        label_confidence = label_confidence.cuda()
        label_start = label_start.cuda()
        label_end = label_end.cuda()
        if torch.sum((label_confidence > 0.7).float()) == 0:
            continue
        # N,2,D,T   N,T    N,T
        conf, start, end, loss_conns = model(input_data)
        loss_list = criterion(conf, start, end, loss_conns, label_confidence, label_start, label_end)
        optimizer.zero_grad()
        loss_list[0].backward()
        optimizer.step()

    epoch_b_loss += loss_list[1].cpu().detach().numpy()
    epoch_prop_com_loss += loss_list[2].cpu().detach().numpy()
    epoch_prop_cls_loss += loss_list[3].cpu().detach().numpy()
    epoch_norm_loss += loss_list[-1].cpu().detach().numpy()
    epoch_loss += loss_list[0].cpu().detach().numpy()

    print(
        "training loss ( [epoch %d] boundary_loss: %.5f, prop_cls_loss:%.5f, prop_com_loss:%.5f, norm_loss:%.5f, total_loss:%.5f )" %
        (
            epoch, epoch_b_loss / (iter + 1), epoch_prop_cls_loss / (iter + 1), epoch_prop_com_loss / (iter + 1),
            epoch_norm_loss / (iter + 1), epoch_loss / (iter + 1)
        )
    )


def test(args, data_loader, model, criterion, epoch):
    model.eval()
    criterion.eval()

    epoch_prop_com_loss = 0
    epoch_prop_cls_loss = 0
    epoch_b_loss = 0
    epoch_loss = 0
    epoch_norm_loss = 0
    loss_list = []
    for iter, (input_data, label_confidence, label_start, label_end) in enumerate(data_loader):
        input_data = input_data.cuda()
        label_confidence = label_confidence.cuda()
        label_start = label_start.cuda()
        label_end = label_end.cuda()
        # N, 2, D, T   N, T    N, T
        if torch.sum((label_confidence > 0.7).float()) == 0:
            continue
        confidence_map, start, end, loss_G = model(input_data)
        loss_list = criterion(confidence_map, start, end, loss_G, label_confidence, label_start, label_end)

    epoch_b_loss += loss_list[1].cpu().detach().numpy()
    epoch_prop_com_loss += loss_list[2].cpu().detach().numpy()
    epoch_prop_cls_loss += loss_list[3].cpu().detach().numpy()
    epoch_norm_loss += loss_list[-1].cpu().detach().numpy()
    epoch_loss += loss_list[0].cpu().detach().numpy()

    print(
        "testing loss ( [epoch %d] boundary_loss: %.5f, prop_cls_loss:%.5f, prop_com_loss:%.5f, norm_loss:%.5f, total_loss:%.5f )" %
        (
            epoch, epoch_b_loss / (iter + 1), epoch_prop_cls_loss / (iter + 1), epoch_prop_com_loss / (iter + 1),
            epoch_norm_loss / (iter + 1), epoch_loss / (iter + 1)
        )
    )
    state = {'epoch': epoch + 1,
             'state_dict': model.state_dict()}

    torch.save(state, os.path.join(args["checkpoint_path"], "PRSA_checkpoint_%d.pth.tar" % (epoch)))
    if epoch_loss < model.module.best_loss:
        model.module.best_loss = epoch_loss
        torch.save(state, os.path.join(args["checkpoint_path"], "PRSA_checkpoint_%d.pth.tar" % (epoch)))
