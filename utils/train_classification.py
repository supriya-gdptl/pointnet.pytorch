from __future__ import print_function
import argparse
import os
import random
import time
import torch.optim as optim
import torch.utils.data
from pointnet.dataset import ShapeNetDataset, ModelNetDataset, HDF5_ModelNetDataset
from pointnet.model import PointNetCls, feature_transform_regularizer
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

from misc.common_functions import write_loss_acc, write_log


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--batchSize', type=int, default=32, help='input batch size')
    parser.add_argument(
        '--num_points', type=int, default=2500, help='number of points in point cloud')
    parser.add_argument(
        '--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument(
        '--nepoch', type=int, default=250, help='number of epochs to train for')
    parser.add_argument('--outf', type=str, default='cls', help='output folder')
    parser.add_argument('--model', type=str, default='', help='model path')
    parser.add_argument('--dataset', type=str, required=True, help="dataset path")
    parser.add_argument('--dataset_type', type=str, default='shapenet', help="dataset type shapenet|modelnet40")
    parser.add_argument('--feature_transform', action='store_true', help="use feature transform")

    opt = parser.parse_args()

    # create output folder if it does not exists
    os.makedirs(opt.outf, exist_ok=True)

    # log
    logtime = time.strftime("%d_%m_%Y_%H_%M_%S")
    logfile = open(os.path.join(opt.outf, f'log_{logtime}.txt'), 'w')
    loss_acc_file = open(os.path.join(opt.outf, f'loss_acc_{logtime}.txt'), 'w')
    loss_acc_file.write("mode,epoch,loss,acc\n")
    print(f"Saving log file to: {os.path.abspath(logfile.name)}")

    write_log(logfile=logfile,msg=opt)

    # run misc/common_functions.py to replace model file extensions from .ply to .off in {train/val/test/trainval}.txt

    opt.manualSeed = random.randint(1, 10000)  # fix seed
    write_log(logfile=logfile,msg=f"Random Seed: {opt.manualSeed}")
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    if opt.dataset_type == 'shapenet':
        dataset = ShapeNetDataset(
            root=opt.dataset,
            classification=True,
            npoints=opt.num_points)

        test_dataset = ShapeNetDataset(
            root=opt.dataset,
            classification=True,
            split='test',
            npoints=opt.num_points,
            data_augmentation=False)
    elif opt.dataset_type == 'modelnet40':
        dataset = ModelNetDataset(
            root=opt.dataset,
            npoints=opt.num_points,
            split='trainval')

        test_dataset = ModelNetDataset(
            root=opt.dataset,
            split='test',
            npoints=opt.num_points,
            data_augmentation=False)
    elif opt.dataset_type == 'hdf5_modelnet40':
        dataset = HDF5_ModelNetDataset(
            root=opt.dataset,
            npoints=opt.num_points,
            split='train')

        test_dataset = HDF5_ModelNetDataset(
            root=opt.dataset,
            split='test',
            npoints=opt.num_points,
            data_augmentation=False)
    else:
        exit('wrong dataset type')

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batchSize,
        shuffle=True,
        num_workers=int(opt.workers))

    testdataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=opt.batchSize,
            shuffle=True,
            num_workers=int(opt.workers))

    write_log(logfile=logfile,
              msg=f"number of training examples: {len(dataset)}\nnumber of test example: {len(test_dataset)}")
    num_classes = len(dataset.classes)
    write_log(logfile=logfile,msg=f'number of classes {num_classes}')

    classifier = PointNetCls(k=num_classes, feature_transform=opt.feature_transform)

    if opt.model != '':
        classifier.load_state_dict(torch.load(opt.model))

    optimizer = optim.Adam(classifier.parameters(), lr=0.001, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    classifier = classifier.cuda()

    num_batch = len(dataset) / opt.batchSize

    for epoch in range(opt.nepoch):

        scheduler.step()
        train_loss = []
        train_acc = []

        for i, data in enumerate(dataloader, 0):
            points, target = data
            target = target[:, 0]
            points = points.transpose(2, 1)
            points, target = points.cuda(), target.cuda()
            optimizer.zero_grad()
            classifier = classifier.train()
            pred, trans, trans_feat = classifier(points)
            loss = F.nll_loss(pred, target)
            if opt.feature_transform:
                loss += feature_transform_regularizer(trans_feat) * 0.001
            loss.backward()
            optimizer.step()
            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(target.data).cpu().sum()

            train_loss.append(loss.item())
            train_acc.append(correct.item()/float(opt.batchSize))

            write_log(logfile=logfile,
                      msg=f'[{epoch}: {i}/{num_batch}] train loss: {loss.item()}, '
                          f'accuracy: {correct.item()/float(opt.batchSize)}')

        write_log(logfile=logfile,
                  msg=f'\nTRAIN [{epoch}/{opt.nepoch}] train loss: {np.array(train_loss).mean()}, '
                      f'accuracy: {np.array(train_acc).mean()}\n')

        write_loss_acc(loss_acc_file=loss_acc_file, mode='train',
                       loss=np.array(train_loss).mean(), acc=np.array(train_acc).mean(), epoch=epoch)

        # test on test data after one epoch
        test_loss = []
        test_acc = []
        for j, testdata in enumerate(testdataloader, 0):
            points, target = testdata
            target = target[:, 0]
            points = points.transpose(2, 1)
            points, target = points.cuda(), target.cuda()
            classifier = classifier.eval()
            pred, _, _ = classifier(points)
            loss = F.nll_loss(pred, target)
            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(target.data).cpu().sum()

            test_loss.append(loss.item())
            test_acc.append(correct.item()/float(opt.batchSize))

        write_log(logfile=logfile,
                  msg=f'\nTEST [{epoch}/{opt.nepoch}] loss: {np.array(test_loss).mean()}, '
                      f'accuracy: {np.array(test_acc).mean()}\n')
        write_loss_acc(loss_acc_file=loss_acc_file, mode='test',
                       loss=np.array(test_loss).mean(), acc=np.array(test_acc).mean(), epoch=epoch)

        torch.save(classifier.state_dict(), '%s/cls_model_%d.pth' % (opt.outf, epoch))

    total_correct = 0
    total_testset = 0
    for i,data in tqdm(enumerate(testdataloader, 0)):
        points, target = data
        target = target[:, 0]
        points = points.transpose(2, 1)
        points, target = points.cuda(), target.cuda()
        classifier = classifier.eval()
        pred, _, _ = classifier(points)
        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(target.data).cpu().sum()
        total_correct += correct.item()
        total_testset += points.size()[0]

    write_log(logfile=logfile,msg=f"final accuracy {total_correct / float(total_testset)}")

