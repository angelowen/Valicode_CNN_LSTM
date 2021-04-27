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
DATASET_ROOT = './train'
Test_ROOT = './test'

def train():
    data_transform = transforms.Compose([
        # transforms.RandomResizedCrop(20),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.456], std=[0.224])
    ])
    train_set = LabelDataset(Path(DATASET_ROOT), data_transform)
    data_loader = DataLoader(dataset=train_set, batch_size=4, shuffle=True, num_workers=1)

    model = Combine()
    model = model.cuda(CUDA_DEVICES)
    model.train()

    best_model_params = copy.deepcopy(model.state_dict())
    best = 0.0
    num_epochs = 200
    criterion = nn.CrossEntropyLoss() #criterion = nn.MSELoss()
    # optimizer = torch.optim.SGD(params=model.parameters(), lr=0.001, momentum=0.9)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001, betas=(0.9, 0.999))

    for epoch in range(num_epochs):
        print(f'Epoch: {epoch + 1}/{num_epochs}')
        print('-' * len(f'Epoch: {epoch + 1}/{num_epochs}'))

        training_loss = 0.0
        training_corrects = 0

        for i, (inputs, labels) in enumerate(data_loader):
            # labels = torch.Tensor([list(item) for item in labels])
            labels=np.array(labels).astype(int)
            labels=torch.from_numpy(labels)
            # labels=torch.tensor(labels,dtype=torch.int64)

            labels = labels.type(torch.LongTensor)

            
            inputs = Variable(inputs.cuda(CUDA_DEVICES))
            labels = Variable(labels.cuda(CUDA_DEVICES))

            optimizer.zero_grad()

            outputs = model(inputs)
            # print("output size: " ,outputs.shape)
            _, preds = torch.max(outputs.data, 1)
            # print("pred size: ",preds.shape)

            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            training_loss += loss.item() * inputs.size(0)
            # training_corrects += torch.sum(preds == labels.data) ##revise

        training_loss = training_loss / len(train_set)
        # training_acc = training_corrects.double() / len(train_set) 

        print(f'Training loss: {training_loss:.4f}\n')

        if training_loss > best:
            best = training_loss
            best_model_params = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_model_params)
    torch.save(model, f'model-{best:.02f}-best_train_acc.pth')
    return model


def test(model):
    data_transform = transforms.Compose([
        # transforms.RandomResizedCrop(20),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.456], std=[0.224])
    ])
    dataset_root = Path(Test_ROOT)

    # model = torch.load(model)
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
    model = train()
    test(model)