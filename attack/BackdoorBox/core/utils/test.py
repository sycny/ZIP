import os
import os.path as osp
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, MNIST, DatasetFolder

from .accuracy import accuracy
from .log import Log
from settings import base_args, base_config
args, config = base_args, base_config

if args.useAVGUP:
     print("===============================Using the Avg_up defense method==========================")

def MeanUpsample(x, scale):
    n, c, h, w = x.shape
    out = torch.zeros(n, c, h, scale, w, scale).to(x.device) + x.view(n,c,h,1,w,1)
    out = out.view(n, c, scale*h, scale*w)
    return out

def preprocess(data, deg_scale):
    
    scale=round(deg_scale)
    A = torch.nn.AdaptiveAvgPool2d((data.shape[-1]//scale,data.shape[-1]//scale))
    y = A(data)
    Ap = lambda z: MeanUpsample(z,scale)
    Apy = Ap(y)
    
    return Apy


def _seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _test(model, dataset, device, batch_size=16, num_workers=8):
    with torch.no_grad():
        test_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False,
            pin_memory=True,
            worker_init_fn=_seed_worker
        )

        model = model.to(device)
        #model = nn.DataParallel(model, device_ids=args.gpulist)
        model.eval()

        predict_digits = []
        labels = []
        for batch in test_loader:
            batch_img, batch_label = batch
            batch_img = batch_img.to(device)
            if args.useAVGUP:
                 batch_img = preprocess(batch_img, args.deg_scale)
            batch_img = model(batch_img)
            batch_img = batch_img.cpu()
            predict_digits.append(batch_img)
            labels.append(batch_label)

        predict_digits = torch.cat(predict_digits, dim=0)
        labels = torch.cat(labels, dim=0)
        return predict_digits, labels
    
def _pois_test(model, dataset, device, batch_size=16, num_workers=8, label_type = 'label_orig'):
    with torch.no_grad():
        test_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False,
            pin_memory=True,
            worker_init_fn=_seed_worker
        )

        model = model.to(device)
        #model = nn.DataParallel(model, device_ids=args.gpulist)
        model.eval()

        predict_digits = []
        labels = []
        for batch in test_loader:
            batch_img, batch_label = batch
            batch_img = batch_img.to(device)
            if args.useAVGUP:
                 batch_img = preprocess(batch_img, args.deg_scale)
            batch_img = model(batch_img)
            batch_img = batch_img.cpu()
            predict_digits.append(batch_img)
            labels.append(batch_label[label_type])

        predict_digits = torch.cat(predict_digits, dim=0)
        labels = torch.cat(labels, dim=0)
        return predict_digits, labels


def test(model, dataset, schedule):

    if 'device' in schedule and schedule['device'] == 'GPU':
        if 'CUDA_VISIBLE_DEVICES' in schedule:
            os.environ['CUDA_VISIBLE_DEVICES'] = schedule['CUDA_VISIBLE_DEVICES']

        assert torch.cuda.device_count() > 0, 'This machine has no cuda devices!'
        assert schedule['GPU_num'] >0, 'GPU_num should be a positive integer'
        print(f"This machine has {torch.cuda.device_count()} cuda devices, and use {schedule['GPU_num']} of them to train.")

        if schedule['GPU_num'] == 1:
            device = torch.device("cuda:0")
        else:
            device = torch.device("cuda")
            model = nn.DataParallel(model, device_ids=args.gpulist)

    # Use CPU
    else:
        device = torch.device("cpu")
        
    work_dir = osp.join(schedule['save_dir'], schedule['experiment_name'] + '_' + time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime()))
    os.makedirs(work_dir, exist_ok=True)
    log = Log(osp.join(work_dir, 'log.txt'))
    last_time = time.time()

    if schedule['metric'] == 'BA':
         predict_digits, labels = _test(model, dataset, device, schedule['batch_size'], schedule['num_workers'])
         print('BA_lables', labels)
    elif schedule['metric'] == 'ASR':
         predict_digits, labels = _pois_test(model, dataset, device, schedule['batch_size'], schedule['num_workers'], label_type='label_pois')
         print('ASR_lables', labels)
    elif schedule['metric'] == 'PA':
         predict_digits, labels = _pois_test(model, dataset, device, schedule['batch_size'], schedule['num_workers'], label_type='label_orig')
         print('PA_lables', labels)    
              
    #predict_digits, labels = _test(model, dataset, device, schedule['batch_size'], schedule['num_workers'])
    total_num = labels.size(0)
    prec1, prec5 = accuracy(predict_digits, labels, topk=(1, 5))
    top1_correct = int(round(prec1.item() / 100.0 * total_num))
    top5_correct = int(round(prec5.item() / 100.0 * total_num))
    msg = f"==========Test result on {schedule['metric']}==========\n" + \
            time.strftime("[%Y-%m-%d_%H:%M:%S] ", time.localtime()) + \
            f"Top-1 correct / Total: {top1_correct}/{total_num}, Top-1 accuracy: {top1_correct/total_num}, Top-5 correct / Total: {top5_correct}/{total_num}, Top-5 accuracy: {top5_correct/total_num}, time: {time.time()-last_time}\n"
    log(msg)
