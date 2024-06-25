import numpy as np
import torch

from model.feature_extractor import UncertaintyDropout


class DummyModel(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.lin1 = torch.nn.Linear(10, 20)
        self.relu1 = torch.nn.ReLU()
        #self.drop = UncertaintyDropout(0.5)
        self.drop = torch.nn.Dropout(0.5)

    def forward(self, x):
        return self.drop(self.relu1(self.lin1(x)))


def main():
    input = torch.Tensor(np.ones((1, 1, 1, 10)))
    net = DummyModel()
    net.eval()
    for _ in range(10):
        out = net(input)
        print(list(out))



if __name__ == "__main__":
    main()