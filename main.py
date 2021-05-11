import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from dataloader.emnist import Emnist
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms, models
from gan.generator import generator
from gan.discriminator import discriminator
import argparse
from tqdm import tqdm
import os
from PIL import Image
import PIL

# nets with cifar https://github.com/chenyaofo/pytorch-cifar-models
import pytorch_cifar_models.pytorch_cifar_models.resnet as resnet

# nets with cifar https://github.com/kuangliu/pytorch-cifar
import pytorch_cifar
from utils import *
from utils2 import *

def forward_discriminator(gen, disc, data, y, criterion):
    mini_batch = data.size()[0]
    D_real = torch.ones((mini_batch,1))#.cuda()
    D_fake = torch.zeros((mini_batch,1))#.cuda()

    D_result = disc(data)
    D_real_loss = criterion(D_result, D_real)

    #noise = torch.randn((mini_batch, 10))#.cuda()
    G_sample = gen(y)
    D_result = disc(G_sample)

    D_fake_loss = criterion(D_result, D_fake)

    D_train_loss = D_real_loss + D_fake_loss

    return D_train_loss

def eval_model(gen):
    test = []
    for i in range(14):
        arr = np.zeros((14,))
        arr[i] = 1.
        test.append(arr)

    test = torch.from_numpy(np.array(test))

    out = gen(test)
    out = out.view(-1, 1, 28, 28)
    imshow(torchvision.utils.make_grid(out.detach()))
        
def forward_generator(gen, disc, data, y, criterion, crit):
    mini_batch = data.size()[0]
    #noise = torch.randn((mini_batch, 10))#.cuda()
    out = torch.ones((mini_batch,1)).cuda()

    G_result = gen(y)
    D_result = disc(G_result)
    G_train_loss = criterion(D_result, out)
    G_real_loss = crit(G_result, data)

    G_loss = 0.1 * G_train_loss + G_real_loss

    return G_loss

def train_gan(epochs, lr, dataset_dir):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=0.5, std=0.5)
    ])

    train_loader = torch.utils.data.DataLoader(Emnist(root_dir='./data/mnist', transforms=transform), batch_size=8)
    print('here')

    gen = generator(input_size=14, n_class=28*28)
    #gen.cuda()
    disc = discriminator(input_size=28*28, n_class=1)
    #disc#.cuda()

    gen_optimizer = optim.Adam(gen.parameters(), lr=lr)
    disc_optimizer = optim.Adam(disc.parameters(), lr=lr)

    criterion = torch.nn.BCELoss()
    crit = torch.nn.MSELoss()

    last_d = np.inf
    last_g = np.inf

    for epoch in tqdm(range(epochs)):
        for load_data, y in train_loader:
            #imshow(torchvision.utils.make_grid(load_data), f'{y}')

            load_data = load_data.view(-1, 28 * 28)#.cuda()
            disc.zero_grad()

            if last_d > 0.2 * last_g:
                D_train_loss = forward_discriminator(gen, disc, load_data, y, criterion)

                D_train_loss.backward()
                disc_optimizer.step()

                last_d = D_train_loss.item()

            gen.zero_grad()
            G_train_loss = forward_generator(gen, disc, load_data, y, criterion, crit)
            G_train_loss.backward()
            gen_optimizer.step()

            last_g = G_train_loss.item()

            
            print(f'[{epoch+1}/{epochs}]: loss_d: {last_d}, loss_g: {last_g}', end='\r')

        if epoch % 500 == 0:
            eval_model(gen)

def classify_images():
    preds = []
    model = getattr(resnet, f"cifar100_resnet32")()

    checkpoint = torch.load('cifar_mode/cifar100_resnet20-23dac2f1.pt')
    model.load_state_dict(checkpoint)
    model.eval()

    transform = transform.Compose([transform.ToTensor()])
    dataset = torchvision.datasets.CIFAR100("dataset/cifar", train=True, transforms=transforms, download=True)

    train_loader = torch.utils.data.DataLoader(dataset, batch_len=8)

    files = os.listdir('./images/infer')
    
    for filename in sorted(files):
        img = transform(PIL.Image.open(filename))
        
        out = model(x)
        out = torch.argmax(out, dim=-1)

        pred = out.item()
        label = cifar_labels['fine_label_names'][pred]
        preds.append([pred, label])
        print(f'{filename} -> prediction [{pred}: {label}]')


    # for x, y in train_loader:
    #     x = torch.unsqueeze(img, dim=0)
    #     out = model(x)

    #     out = torch.argmax(out, dim=-1)

    #     for i in range(len(out)):
    #         torchvision.utils.save_image(x[i], f"infer_images/{out[i]}.png")
    #         input()

    return preds

def generate_message_from_encode(encode):
    encode_pred = [ pred for pred, label in encode ]
    print(f'\n{encode_pred}')
    gen = generator(input_size=14, n_class=28*28)
    checkpoint = torch.load('./gan_pretrainedgenerator_emnist.pt', map_location=torch.device('cpu'))

    gen.load_state_dict(checkpoint['model_state_dict'])
    one_hot = torch.from_numpy(num_to_one_hot_vector(encode_pred))

    out = gen(one_hot)
    out = out.view(-1, 1, 28, 28)
    imshow(torchvision.utils.make_grid(out.detach()))

def decode_message_from_images():
    print('Predicting images class with ResNet20...')
    encode = classify_images()

    print('\nOutput predictions: ', encode)

    encode_num = encode_label_from_pred(encode)
    encode_to_one_hot(encode_num)
    a = np.array(encode_num)
    encode_one_hot = np.zeros((a.size, a.max()+1))
    encode_one_hot[np.arange(a.size),a] = 1

    generate_message_from_encode(encode)

parser = argparse.ArgumentParser(description="GAN")
parser.add_argument("--dataset_dir", type=str, default="EMNIST",
                    help="which dataset")
parser.add_argument("--num_epochs", type=int, default=10,
                    help="number of epochs to train (default: 10)")
parser.add_argument("--lr", type=float, default=0.0002,
                    help="learning rate for training (default: 0.0002)")
args = parser.parse_args()

dataset_dir = args.dataset_dir
epochs = args.num_epochs
lr = args.lr

train_gan(epochs, lr, dataset_dir)
decode_message_from_images()
    

            

