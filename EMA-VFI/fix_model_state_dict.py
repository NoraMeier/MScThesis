import torch
import argparse

from Trainer import Model
from model.flow_estimation import UncertaintyDropout2D

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="ckpt/drop_flowest/ours_small_0.pkl", type=str)

    args = parser.parse_args()

    drop = UncertaintyDropout2D()

    model1 = Model(-1, dropout_flowest=True)
    model2 = Model(-1)

    print(set(model1.net.state_dict().keys()) - set(model2.net.state_dict().keys()))

    state_dict = torch.load(args.model_path)



if __name__ == "__main__":
    main()