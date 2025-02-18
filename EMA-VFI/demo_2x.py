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

UNCERTAINTY_SAMPLES = 10
HIGH_UNCERTAINTY_POINT = 33.0 

class Ensemble():
    def __init__(self, model_path):
        self.models = []
        model_dir = os.listdir(model_path)
        for file in model_dir:
            if 'ours_small' in file and file[-4:] == '.pkl':
                print(f"Loading model: {file.split('.')[0]}")
                model = Model(-1)
                model.load_model(file.split('.')[0], custom=True)
                model.eval()
                model.device()
                self.models.append(model)
        print(f"Loaded {len(self.models)} models")

    def predict(self, shape, I0_, I2_, padder, TTA=False):
        preds = np.zeros((len(self.models),) + shape)
        for idx, model in enumerate(self.models):
            preds[idx] = (padder.unpad(model.inference(I0_, I2_, TTA=TTA, fast_TTA=TTA))[0].detach().cpu().numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)
        pred = np.mean(preds, axis=0)
        sd = np.std(preds, axis=0)
        return pred, sd

        
def transform_image(img):
    return (torch.tensor(img.transpose(2, 0, 1)).cuda() / 255.).unsqueeze(0)

def predict_frame_dropout(model, shape, I0_, I2_, padder, n_samples=UNCERTAINTY_SAMPLES, TTA=False):
    preds = np.zeros((n_samples,) + shape)
    for i in range(n_samples):
        preds[i] = (padder.unpad(model.inference(I0_, I2_, TTA=TTA, fast_TTA=TTA))[0].detach().cpu().numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)
    pred = np.mean(preds, axis=0)
    sd = np.std(preds, axis=0)
    return pred, sd

def colormap(sd, map):
    sd_relative = 255 * sd / sd.max()
    sd_absolute = 255 * sd / HIGH_UNCERTAINTY_POINT
    sd_relative = sd_relative.astype(np.uint8)
    sd_absolute = sd_absolute.astype(np.uint8)
    sd_rel_color = cv2.cvtColor(sd_relative, cv2.COLOR_RGB2GRAY)
    sd_abs_color = cv2.cvtColor(sd_absolute, cv2.COLOR_RGB2GRAY)
    sd_rel_color = cv2.applyColorMap(sd_rel_color, map)
    sd_abs_color = cv2.applyColorMap(sd_abs_color, map)
    return sd_rel_color, sd_abs_color

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='ours_small', type=str)
parser.add_argument('--input_dir', default='example/ex1', type=str)
parser.add_argument('--uncertainty', default="none", type=str, choices=["none", "featextr", "flowest", "refine", "ensemble"])
parser.add_argument('--video', default='None', type=str, choices=['None', 'catvideo', 'bg3', 'jojos', 'vimeo'])

args = parser.parse_args()
assert args.model in ['ours', 'ours_small'], 'Model not exists!'


'''==========Model setting=========='''
TTA = True
if args.model == 'ours_small':
    TTA = False
    cfg.MODEL_CONFIG['LOGNAME'] = 'ours_small'
    cfg.MODEL_CONFIG['MODEL_ARCH'] = cfg.init_model_config(
        F = 16,
        depth = [2, 2, 2, 2, 2]
    )
else:
    cfg.MODEL_CONFIG['LOGNAME'] = 'ours'
    cfg.MODEL_CONFIG['MODEL_ARCH'] = cfg.init_model_config(
        F = 32,
        depth = [2, 2, 2, 4, 4]
    )
model = Model(-1)
model.load_model(name=args.model, custom=True)
model.eval()
model.device()


print(f'=========================Start Generating=========================')
I0 = cv2.imread(args.input_dir + '/img1.jpg')
I2 = cv2.imread(args.input_dir + '/img3.jpg')

I0_ = transform_image(I0)
I2_ = transform_image(I2)

padder = InputPadder(I0_.shape, divisor=32)
I0_, I2_ = padder.pad(I0_, I2_)
    
mid = (padder.unpad(model.inference(I0_, I2_, TTA=TTA, fast_TTA=TTA))[0].detach().cpu().numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)

no_uncertainty = False
if args.uncertainty == "none":
    no_uncertainty = True
    if args.video == "None":
        cv2.imwrite(args.input_dir + '/img2.jpg', mid)
        exit(0)
elif args.uncertainty == "ensemble":
    ensemble = Ensemble('ckpt')
else:
    model_name = args.model
    if args.uncertainty == "featextr":
        model = Model(-1, dropout_featextr=True)
    elif args.uncertainty == "flowest":
        model = Model(-1, dropout_flowest=True)
        model_name = f"drop_flowest/{args.model}_0"
    elif args.uncertainty == "refine":
        model = Model(-1, dropout_refine=True)
        model_name = f"drop_refine/{args.model}_0"
    else:
        print("uncertainty method not implemented")
        exit(-1)
    model.load_model(name=model_name, custom=True)
    model.eval()
    model.device()

if args.video != "None":
    if not os.path.exists(f'video/{args.video}/interpolated_frames_{args.uncertainty}'):
        os.mkdir(f'video/{args.video}/interpolated_frames_{args.uncertainty}')
    if args.video == "vimeo":
        video_path = '../vimeo_dataset/vimeo_triplet/sequences/00001'
        I0 = cv2.imread(f"{video_path}/0001/im1.png")
        I0_ = transform_image(I0)
        padder = InputPadder(I0_.shape, divisor=32)

        triplet_paths = os.listdir(video_path)
        for idx, triplet in enumerate(triplet_paths):
            I0 = cv2.imread(f"{video_path}/{triplet}/im1.png")
            I2 = cv2.imread(f"{video_path}/{triplet}/im3.png")
            gt = cv2.imread(f"{video_path}/{triplet}/im2.png")
            I0_ = padder.pad(transform_image(I0))[0]
            I2_ = padder.pad(transform_image(I2))[0]
            if args.uncertainty == "none":
                pred = (padder.unpad(model.inference(I0_, I2_, TTA=TTA, fast_TTA=TTA))[0].detach().cpu().numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)
                final_img = np.concatenate((gt, pred), axis=1)
            else:
                if args.uncertainty == "ensemble":
                    pred, sd = ensemble.predict(I0.shape, I0_, I2_, padder)
                else:
                    pred, sd = predict_frame_dropout(model, I0.shape, I0_, I2_, padder)
                sd_rel, sd_abs = colormap(sd, cv2.COLORMAP_OCEAN)
                img_height, img_width = pred.shape[0], pred.shape[1]
                if img_height > 2 * img_width:
                    final_img = np.concatenate((gt, pred, sd_rel, sd_abs), axis=1)
                else:
                    final_img = np.concatenate(
                        (
                            np.concatenate((gt, pred), axis=1),
                            np.concatenate((sd_rel, sd_abs), axis=1)
                        ), 
                        axis=0
                    )
            cv2.imwrite(f"video/{args.video}/interpolated_frames_{args.uncertainty}/frame_{idx}.jpg", final_img)
    else:
        video_path = f'video/{args.video}/{args.video}.mp4'
        vidcap = cv2.VideoCapture(video_path)
        _, I0 = vidcap.read()
        _, gt = vidcap.read()
        success, I2 = vidcap.read()
        if success:
            I0_ = transform_image(I0)
        else:
            print("Select a video with more than 2 frames!")
            exit()
        padder = InputPadder(I0_.shape, divisor=32)
        I0_ = padder.pad(I0_)[0]
        idx = 0

        print("==================Starting Video Interpolation=================")
        while success:
            I2_ = transform_image(I2)
            I2_ = padder.pad(I2_)[0]
            if no_uncertainty:
                pred = (padder.unpad(model.inference(I0_, I2_, TTA=TTA, fast_TTA=TTA))[0].detach().cpu().numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)
                final_img = np.concatenate((gt, pred), axis=1)
            else:
                if args.uncertainty == "ensemble":
                    pred, sd = ensemble.predict(I0.shape, I0_, I2_, padder)
                else:
                    pred, sd = predict_frame_dropout(model, I0.shape, I0_, I2_, padder)
                sd_rel, sd_abs = colormap(sd, cv2.COLORMAP_OCEAN)
                img_height, img_width = pred.shape[0], pred.shape[1]
                if img_height > 2 * img_width:
                    final_img = np.concatenate((gt, pred, sd_rel, sd_abs), axis=1)
                else:
                    final_img = np.concatenate(
                        (
                            np.concatenate((gt, pred), axis=1),
                            np.concatenate((sd_rel, sd_abs), axis=1)
                        ), 
                        axis=0
                    )
            cv2.imwrite(f"video/{args.video}/interpolated_frames_{args.uncertainty}/frame_{idx}.jpg", final_img)

            idx += 1
            I0_ = I2_
            _, gt = vidcap.read()
            success, I2 = vidcap.read()
else:
    pred, sd = predict_frame_dropout(model, mid.shape, I0_, I2_, padder)
    if (sd == np.zeros(sd.shape)).all():
        print("Problems :(")
    cv2.imwrite(args.input_dir + '/img2.jpg', pred)
    sd_rel, sd_abs = colormap(sd, cv2.COLORMAP_OCEAN)
    cv2.imwrite(args.input_dir + '/std_rel.jpg', sd_rel)
    cv2.imwrite(args.input_dir + '/std_abs.jpg', sd_abs)
    print(sd.max())
    

print(f'=========================Done=========================')