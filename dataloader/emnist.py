from torch.utils.data import Dataset
import PIL
from PIL import Image
import numpy as np
import os

class Emnist(Dataset):
    def __init__(self, root_dir, transforms):
        self.root_dir = root_dir
        self.transforms = transforms
        self.letters = []
        
        count = 0
        for img in os.listdir(self.root_dir):
            lbl = np.zeros((14,))
            lbl[count] = 1.
            count += 1
            img = Image.open(f'{self.root_dir}/{img}').convert('L')
            img = PIL.ImageOps.invert(img)
            self.letters.append((np.asarray(img), lbl))

    def __len__(self):
        return len(self.letters)

    def __getitem__(self, idx):
        return self.transforms(self.letters[idx][0]), self.letters[idx][1]
