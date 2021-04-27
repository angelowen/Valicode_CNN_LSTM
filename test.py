import torch
import torch.nn as nn
from models import Combine
from dataset import LabelDataset
from utils import parse_args
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from pathlib import Path
import copy
import numpy as np
from PIL import Image
import glob


args = parse_args()
CUDA_DEVICES = 0
Test_ROOT = './test'
PATH_TO_WEIGHTS = './model-0.93-best_train_acc.pth'

def test():
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.456], std=[0.224])
    ])
    dataset_root = Path(Test_ROOT)

    model = torch.load(PATH_TO_WEIGHTS)
    model = model.cuda(CUDA_DEVICES)
    model.eval()

    for i, filename in enumerate(glob.glob(f'{Test_ROOT}/*')):
        image = Image.open(filename)
        image = data_transform(image).unsqueeze(0)
        inputs = Variable(image.cuda(CUDA_DEVICES))
        outputs = model(inputs)
        _, preds = torch.max(outputs.data, 1)
        print(preds)





if __name__ == '__main__':
    test()