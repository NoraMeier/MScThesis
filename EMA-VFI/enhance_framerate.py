import numpy as np
import cv2
import sys
import torch
import argparse
import os

'''==========import from our code=========='''
sys.path.append('.')
import config as cfg
from Trainer import Model
from benchmark.utils.padder import InputPadder

def transform_image(img):
    return (torch.tensor(img.transpose(2, 0, 1)).cuda() / 255.).unsqueeze(0)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', default='None', type=str, choices=['None', 'catvideo', 'bg3', 'jojos'])
    args = parser.parse_args()

    TTA = False
    cfg.MODEL_CONFIG['LOGNAME'] = 'ours_small'
    cfg.MODEL_CONFIG['MODEL_ARCH'] = cfg.init_model_config(
        F = 16,
        depth = [2, 2, 2, 2, 2]
    )
    model = Model(-1)
    model.load_model(name='ours_small', custom=True)
    model.eval()
    model.device()

    video_path = f'video/{args.video}/{args.video}.mp4'
    vidcap = cv2.VideoCapture(video_path)
    _, I0 = vidcap.read()
    success, I2 = vidcap.read()
    if success:
        I0_ = transform_image(I0)
    else:
        print("Select a video with more than 1 frame!")
        exit()
    idx = 0
    padder = InputPadder(I0_.shape, divisor=32)
    I0_ = padder.pad(I0_)[0]
    
    if not os.path.exists(f'video/{args.video}/high_framerate'):
        os.mkdir(f'video/{args.video}/high_framerate')

    cv2.imwrite(f"video/{args.video}/high_framerate/frame_{idx}.jpg", I0)
    idx += 1

    print("==================Starting Video Interpolation=================")
    while success:
        I2_ = transform_image(I2)
        I2_ = padder.pad(I2_)[0]
        pred = (padder.unpad(model.inference(I0_, I2_, TTA=TTA, fast_TTA=TTA))[0].detach().cpu().numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)
        cv2.imwrite(f"video/{args.video}/high_framerate/frame_{idx}.jpg", pred)
        idx += 1
        cv2.imwrite(f"video/{args.video}/high_framerate/frame_{idx}.jpg", I2)
        idx += 1
        I0_ = I2_
        success, I2 = vidcap.read()


if __name__ == "__main__":
    main()