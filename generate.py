import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms
from generator import generator

def imshow(img, text=None):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    if text is not None:
        plt.title(text)
    plt.show()

if __name__ == "__main__":
    gen = generator(input_size=14, n_class=28*28)
    checkpoint = torch.load('generator_emnist.pt', map_location=torch.device('cpu'))

    gen.load_state_dict(checkpoint['model_state_dict'])

    while True:
        #test_num = [0, 10, 1, 11, 1, 4, 7, 0, 0, 0, 6, 14, 8, 8, 0, 0, 5, 1, 12, 14, 13, 3, 4, 9]
        test = []
        for i in range(len(test_num)):
            arr = np.zeros((14,))

            if test_num[i] != 0:
                arr[test_num[i] - 1] = 1.
            test.append(arr)

        test = torch.from_numpy(np.array(test))

        out = gen(test)
        out = out.view(-1, 1, 28, 28)
        imshow(torchvision.utils.make_grid(out.detach()))
