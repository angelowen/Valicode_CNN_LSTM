import torch.nn as nn
import torch.nn.functional as F
import math
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.features2 = nn.Sequential(
            nn.Linear(in_features=21504, out_features=2048),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(in_features=2048, out_features=1600),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(in_features=1600, out_features=1600)
        )


    def forward(self, x):

        x = self.features(x) # [4, 256, 7, 12]
        x = x.view(x.size(0), -1) # [4, 21504]
        x = self.features2(x) # [4, 1600]
        return x


class Combine(nn.Module):
    def __init__(self):
        super(Combine, self).__init__()
        self.cnn = CNN()
        self.rnn = nn.LSTM(
            input_size=160, 
            hidden_size=64, 
            num_layers=2,
            batch_first=True)
        self.linear = nn.Linear(64,10)
        self.fc = nn.Linear(64, 4) # hidden_size=64, output_size =4

    def forward(self, x):
        batch_size, C, H, W = x.size()
        timesteps = 10
        c_out = self.cnn(x) # [4, 1600]
        r_in = c_out.view(batch_size, timesteps, -1) # rnn_in [4,10,160]
        # print("rnn_in : ",r_in.shape)
        r_out, _ = self.rnn(r_in) # rnn out [4, 10, 64]
        r_out2 = self.fc(r_out) #  [4, 10, 4]
        # print("rnn_out : ",r_out.shape)
       
        return r_out2 #F.log_softmax(r_out2, dim=2)

