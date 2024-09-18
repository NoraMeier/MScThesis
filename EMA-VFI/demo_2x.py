import numpy as np
import cv2
import sys
import torch
import argparse
from imageio import mimsave

'''==========import from our code=========='''
sys.path.append('.')
import config as cfg
from Trainer import Model
from benchmark.utils.padder import InputPadder


class Ensemble():
    def __init__(self, model_path):
        self.models = []
        



UNCERTAINTY_SAMPLES = 10

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='ours', type=str)
parser.add_argument('--input_dir', default='example/ex1', type=str)
parser.add_argument('--uncertainty', default="none", type=str, choices=["none", "featextr", "flowest", "refine", "ensemble"])

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

I0_ = (torch.tensor(I0.transpose(2, 0, 1)).cuda() / 255.).unsqueeze(0)
I2_ = (torch.tensor(I2.transpose(2, 0, 1)).cuda() / 255.).unsqueeze(0)

padder = InputPadder(I0_.shape, divisor=32)
I0_, I2_ = padder.pad(I0_, I2_)
    
mid = (padder.unpad(model.inference(I0_, I2_, TTA=TTA, fast_TTA=TTA))[0].detach().cpu().numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)

if args.uncertainty == "none":
    images = [I0[:, :, ::-1], mid[:, :, ::-1], I2[:, :, ::-1]]
    #mimsave('example/out_2x.gif', images, fps=3)
    cv2.imwrite(args.input_dir + '/img2.jpg', mid)

elif args.uncertainty == "ensemble":
    ensemble = Ensemble()
else:
    model_name = args.model
    if args.uncertainty == "featextr":
        model = Model(-1, dropout_featextr=True)
    elif args.uncertainty == "flowest":
        model = Model(-1, dropout_flowest=True)
        model_name = f"drop_flowest/{args.model}_0"
    elif args.uncertainty == "refine":
        model = Model(-1, )
        model_name = f"drop_refine/{args.model}_0"
    else:
        print("uncertainty method not implemented")
        exit(-1)
    model.load_model(name=model_name, custom=True)
    model.eval()
    model.device()
    
    preds = np.zeros((UNCERTAINTY_SAMPLES,) + mid.shape)
    for i in range(UNCERTAINTY_SAMPLES):
        preds[i] = (padder.unpad(model.inference(I0_, I2_, TTA=TTA, fast_TTA=TTA))[0].detach().cpu().numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)
    pred = np.mean(preds, axis=0)
    sd = np.std(preds, axis=0)
    if (sd == np.zeros(sd.shape)).all():
        print("Problems :(")
    cv2.imwrite(args.input_dir + '/img2.jpg', pred)

    sd_convert = 255 * cv2.cvtColor(sd.astype(np.float32), cv2.COLOR_BGR2GRAY)
    sd_convert = sd_convert.astype(np.int8)
    cv2.imwrite(args.input_dir + '/std.jpg', sd_convert) 
    cv2.imwrite(args.input_dir + '/std_old.jpg', sd) 
    

print(f'=========================Done=========================')