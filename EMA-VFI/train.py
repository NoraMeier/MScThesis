import os
import cv2
import math
import time
import torch
#import torch.distributed as dist
import numpy as np
import random
import argparse

from Trainer import Model
from dataset import VimeoDataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
#from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import RandomSampler
from config import *

device = torch.device("cuda")
exp = os.path.abspath('.').split('/')[-1]

INIT_LR = 2e-4
LR_DECAY = 1e-4
WARM_UP_STEPS = 2500

UNCERTAINTY_FEATURE_EXTRACTION = False
UNCERTAINTY_FLOW_ESTIMATION = False
UNCERTAINTY_REFINE = True

def get_learning_rate(step):
    if step < WARM_UP_STEPS:
        mul = step / WARM_UP_STEPS
        return 2e-4 * mul
    else:
        mul = np.cos((step - WARM_UP_STEPS) / (300 * args.step_per_epoch - WARM_UP_STEPS) * math.pi) * 0.5 + 0.5
        return (2e-4 - 2e-5) * mul + 2e-5

def get_learning_rate_alt(lr, decay):
    lr = lr * (1.0 - decay)
    if lr < 2e-5:
        lr = 2e-5
    return lr


def get_first_epoch(save_dir, log_id):
    step_file_path = f'step_dir/{save_dir}/model{log_id}_step.txt'
    if os.path.exists(step_file_path):
        with open(step_file_path, 'r') as text_file:
            return int(text_file.readline())
    elif not os.path.exists(f'step_dir/{save_dir}'):
        os.mkdir(f'step_dir/{save_dir}')

    return 0


def save_epoch(epoch, save_dir, log_id):
    with open(f'step_dir/{save_dir}/model{log_id}_step.txt', 'w') as file:
        file.write(str(epoch))


def train(model, local_rank, batch_size, data_path, log_id, model_save_dir):
    if local_rank == 0:
        writer = SummaryWriter(f'log/{model_save_dir}train_EMAVFI_{log_id}')
    
    nr_eval = 0
    best = 0
    dataset = VimeoDataset(data_path, train_set=True)
    #sampler = DistributedSampler(dataset)
    sampler = RandomSampler(dataset)
    train_data = DataLoader(dataset, batch_size=batch_size, num_workers=8, pin_memory=True, drop_last=True, sampler=sampler)
    args.step_per_epoch = train_data.__len__()
    dataset_val = VimeoDataset(data_path, train_set=False)
    sampler_val = RandomSampler(dataset_val)
    val_data = DataLoader(dataset_val, batch_size=batch_size, pin_memory=True, num_workers=8, sampler=sampler_val)
    
    start_epoch = get_first_epoch(model_save_dir, log_id)
    if start_epoch > 0:
        model.load_model(name=f"{model_save_dir}/ours_small_{log_id}", custom=True)
    step = start_epoch * args.step_per_epoch

    print('training...')
    time_stamp = time.time()
    for epoch in range(start_epoch, 300):
        save_epoch(epoch, model_save_dir, log_id)
        #sampler.set_epoch(epoch)
        print(f"epoch {epoch}")
        for i, imgs in enumerate(train_data):
            
            data_time_interval = time.time() - time_stamp
            time_stamp = time.time()
            imgs = imgs.to(device, non_blocking=True) / 255.
            imgs, gt = imgs[:, 0:6], imgs[:, 6:]
            learning_rate = get_learning_rate(step)
            _, loss = model.update(imgs, gt, learning_rate, training=True)
            train_time_interval = time.time() - time_stamp
            time_stamp = time.time()
            if step % 200 == 1 and local_rank == 0:
                writer.add_scalar('learning_rate', learning_rate, step)
                writer.add_scalar('loss', loss, step)
            if local_rank == 0:
                print('epoch:{} {}/{} time:{:.2f}+{:.2f} loss:{:.4}'.format(epoch, i, args.step_per_epoch, data_time_interval, train_time_interval, loss))
            step += 1
        nr_eval += 3
        if nr_eval % 3 == 0:
            evaluate(model, val_data, nr_eval, local_rank, log_id, model_save_dir)
        model.save_model(local_rank, directory=model_save_dir)
            
        #dist.barrier()

def evaluate(model, val_data, nr_eval, local_rank, log_id, save_dir):
    if local_rank == 0:
        writer_val = SummaryWriter(f'log/{save_dir}validate_EMAVFI_{log_id}')

    psnr = []
    for _, imgs in enumerate(val_data):
        imgs = imgs.to(device, non_blocking=True) / 255.
        imgs, gt = imgs[:, 0:6], imgs[:, 6:]
        with torch.no_grad():
            pred, _ = model.update(imgs, gt, training=False)
        for j in range(gt.shape[0]):
            psnr.append(-10 * math.log10(((gt[j] - pred[j]) * (gt[j] - pred[j])).mean().cpu().item()))
   
    psnr = np.array(psnr).mean()
    if local_rank == 0:
        print(str(nr_eval), psnr)
        writer_val.add_scalar('psnr', psnr, nr_eval)
        
if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', default=0, type=int, help='local rank')
    parser.add_argument('--world_size', default=4, type=int, help='world size')
    parser.add_argument('--batch_size', default=8, type=int, help='batch size')
    parser.add_argument('--data_path', type=str, help='data path of vimeo90k')
    parser.add_argument('--log_id', default='0', type=str)
    parser.add_argument('--save_dir', default="None", type=str)
    args = parser.parse_args()
    save_dir = f"{args.save_dir}/" if args.save_dir != "None" else ''
        
    #torch.distributed.init_process_group(backend="mpi", world_size=args.world_size)
    torch.cuda.set_device(args.local_rank)
    if args.local_rank == 0 and not os.path.exists('log'):
        os.mkdir('log')
    if save_dir != None and not os.path.exists(f'log/{save_dir[-1]}'):
        os.mkdir(f'log/{save_dir[:-1]}')
    #seed = 1234
    #random.seed(seed)
    #np.random.seed(seed)
    #torch.manual_seed(seed)
    #torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    model = Model(args.local_rank, uncertainty_flowest=UNCERTAINTY_FLOW_ESTIMATION, uncertainty_refine=UNCERTAINTY_REFINE)
    train(model, args.local_rank, args.batch_size, args.data_path, args.log_id, save_dir)
        
