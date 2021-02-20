from __future__ import print_function
import argparse
import torch
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
from pointnet.dataset import HDF5_ModelNetDataset, ModelNetDataset
from pointnet.model import PointNetCls
import torch.nn.functional as F
import trimesh
import os

parser = argparse.ArgumentParser()

parser.add_argument('--model', type=str, default='', help='model path')
parser.add_argument('--feature_transform', action='store_true', help='feature transform')
parser.add_argument('--num_points', type=int, default=1024, help='input batch size')
parser.add_argument('--dataset', type=str, default='', help='dataset path')
parser.add_argument('--output_misclassed_pcd', type=str, required=True, help='folder where misclassed point clouds will be saved')
parser.add_argument('--dataset_type', type=str, default='modelnet40', help='choice=[modelnet40, hdf5_modelnet40]')

opt = parser.parse_args()
print(opt)

os.makedirs(opt.output_misclassed_pcd, exist_ok=True)

if opt.dataset_type == "modelnet40":
    test_dataset = ModelNetDataset(
        root=opt.dataset,
        split='test',
        npoints=opt.num_points,
        data_augmentation=False)
elif opt.dataset_type == "hdf5_modelnet40":
    test_dataset = HDF5_ModelNetDataset(
        root=opt.dataset,
        split='test',
        npoints=opt.num_points,
        data_augmentation=False)
else:
    print('wrong dataset_type')

print('{}: number of test examples:{}'.format(opt.dataset_type, len(test_dataset)))

testdataloader = torch.utils.data.DataLoader(
    test_dataset, batch_size=1)#, shuffle=True)

classifier = PointNetCls(k=len(test_dataset.classes), feature_transform=opt.feature_transform)
classifier.cuda()
classifier.load_state_dict(torch.load(opt.model))
classifier.eval()

class_id2name = {test_dataset.cat[key]: key for key in list(test_dataset.cat.keys())}
failure_cases = dict()
count = 0
for i, data in enumerate(testdataloader, 0):
    points, target, filename = data
    points, target = Variable(points), Variable(target[:, 0])
    points = points.transpose(2, 1)
    points, target = points.cuda(), target.cuda()
    pred, _, _ = classifier(points)
    loss = F.nll_loss(pred, target)

    pred_choice = pred.data.max(1)[1]
    correct = pred_choice.eq(target.data).cpu().sum()
    target = target.item()
    pred_choice = pred_choice.item()
    if correct.item() == 0:
        pts = points.cpu().detach().squeeze(dim=0).numpy().T
        pcd = trimesh.PointCloud(pts)
        name = filename[0].split("/")[-1][:-4]
        pcd.export(os.path.join(opt.output_misclassed_pcd, f"{name}.ply"))

        label_target = class_id2name[target]
        if label_target not in failure_cases:
            failure_cases[label_target] = []
        failure_cases[label_target].append("{}_labelid_{}_name_{}".format(filename[0], pred_choice, class_id2name[pred_choice]))
        count += 1
        # print('example {}: accuracy: {}, target: {}-{}, predicted: {}-{}'.format(filename[0], correct,
        #                                                                          target, label_target,
        #                                                                          pred_choice,
        #                                                                          class_id2name[pred_choice]))
print("*" * 50)
print("Number of failure cases:{}/{}".format(count, len(test_dataset)))
for k in failure_cases:
    print(k + ":")
    print(",\n".join(failure_cases[k]))
    print()
