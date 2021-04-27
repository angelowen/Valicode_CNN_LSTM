from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import pandas as pd


class LabelDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        # self.x = []
        # self.y = []
        self.transform = transform
        df = pd.read_csv('./data.csv',converters = {'answer':str}) ## keep zero in head
        self.x = df.loc[:,'filename'].values.tolist()
        a = df.loc[:,'answer'].values.tolist()
        self.y = []
        for word in a:
            s = [char for char in word]
            self.y.append(s)


    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        image = Image.open(self.x[index])

        if self.transform:
            image = self.transform(image)
        return image, self.y[index]