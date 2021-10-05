#!/usr/bin/env python3
# coding=utf-8

## Zpracování záznamu cílové kamery
# @file CNN_uceni.py
# @brief Učení konvoluční neuronové sítě
# @author Jahoda Vojtěch

__author__ = "Jahoda Vojtěch"
__description__ = "Učení konvoluční neuronové sítě"

import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import PIL
import os
import pandas as pd
import numpy as np
from datetime import datetime
import gzip

from moduls import *


class AthleticDataset(Dataset):
    def __init__(self, pkl_file, transform=None):
        with gzip.open(pkl_file, 'rb') as f:
          df = pd.read_pickle(f)
        self.annotation = pd.DataFrame(df)
        self.transform = transform

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):
        image = self.annotation.iloc[index, 0]
        y_label = torch.tensor([float(self.annotation.iloc[index, 1])])

        if self.transform:
            image = self.transform(image)

        return (image, y_label)


def check_accuracy(loader: torch.utils.data.dataloader.DataLoader, model: 'model CNN', deviation: int = 10,print_eval: bool = False):
    """
    Funkce pro testování přesnosti měření neuronové sítě
    @param loader       -> datová sada na které je proveden test
    @param model        -> model, který se testuje
    @param deviation    -> odchylka chyby testování v ms
    @param print_eval   -> možnost vypsání do cmd výsledku
    @return             -> přesnost modelu nad datovou sadou
    """
    calc_dev = deviation*3/190/2
    num_currect = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)

            scores = model(x)
            num_currect += torch.logical_and(scores <
                                             y+calc_dev, scores > y-calc_dev).sum()
            num_samples += scores.size(0)
        
        if print_eval == True:
            print(f'vzorky {num_currect} / {num_samples} s přesností {float(num_currect)/float(num_samples)*100:.2f}')

    model.train()
    return float(num_currect)/float(num_samples)*100


def save_model(state: 'model CNN', filename: str = 'model.pth.tar', folder: str = 'models'):
    """
    Funkce na ukládání modelu
    @param state        -> model na uložení
    @param filename     -> název souboru pro uložení modelu
    @param models       -> název složky
    """
    if not os.path.exists(folder):
        os.mkdir(folder)
    torch.save(state, os.path.join(folder, filename))


def load_model(model: 'model CNN', filename: str = 'model.pth.tar', folder: str = 'models'):
    """
    Funkce pro možnmost načtení předtrénovaného modelu
    @param model        -> třídy modelu k načtení
    @param filename     -> jméno souboru ve kterém ej model uložen
    @param folder       -> složka uloženého modelu
    @return             -> načtený model ze souboru
    """
    model.load_state_dict(torch.load(os.path.join(folder, filename)))
    return model


def train_model(model: 'trénovaný model CNN',                           #Právě trénovaný model neuronové sítě
                train_loader: torch.utils.data.dataloader.DataLoader,   #datová sada pro trénování
                test_loader: torch.utils.data.dataloader.DataLoader,    #datová sada pro testování
                acc_dict: dict,                                         #slovník, do kterého se ukládají postupné výsledky testování
                lr: float,                                              #učící koeficient
                epochs: int,                                            #počet epoch pro učení
                name: str,                                              #jméno modelu (pojmenování při okládání)
                test_acc: bool = True,                                  #přepínač pro testování
                nth_epoch_test: int = 1,                                #určuje kolikátou epochu se bude provádět testování
                save_models: bool = False                               #ukládání jednotlivých modelů
                ) -> 'natrenovany model':

    #ztrátová funkce MeanSquare
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    max_depth = name
    for epoch in range(epochs):
        #testování přesnosti modelu
        if epoch % nth_epoch_test == 0 and test_acc == True:
            print('['+str(datetime.now().strftime("%H:%M:%S"))+'] ', end='')
            print('epoch', epoch, 'TESTING...', end='')
            acc_dict[max_depth]['train'].append(
                check_accuracy(train_loader, model))
            acc_dict[max_depth]['test'].append(
                check_accuracy(test_loader, model))
            print('train:', round(acc_dict[max_depth]['train'][-1], 2),
                  'test:', round(acc_dict[max_depth]['test'][-1], 2))

            if save_models == True:
                print('SAVING... ', end='')
                save_model(model.state_dict(), filename=str(
                    max_depth)+'_model_'+str(epoch)+'.pth.tar')
                print('model_'+str(epoch)+'.pth.tar DONE')

        print('['+str(datetime.now().strftime("%H:%M:%S"))+'] ', end='')
        print('starting epoch', epoch, '...', end='')

        #učení
        for batch_idx, (data, targets) in enumerate(train_loader):
            data = data.to(device=device)
            targets = targets.to(device=device)

            #forward
            scores = model(data)
            loss = criterion(scores, targets)

            #backward
            optimizer.zero_grad()
            loss.backward()

            #gradient or adam step
            optimizer.step()
        print(' DONE')

    if test_acc == False:
        return model

    epoch = epochs-1
    print('['+str(datetime.now().strftime("%H:%M:%S"))+'] ', end='')
    print('epoch', epoch, 'TESTING...', end='')
    acc_dict[max_depth]['train'].append(check_accuracy(train_loader, model))
    acc_dict[max_depth]['test'].append(check_accuracy(test_loader, model))
    print('train:', round(acc_dict[max_depth]['train'][-1], 2),
          'test:', round(acc_dict[max_depth]['test'][-1], 2))

    if save_models == True:
        print('SAVING... ', end='')
        save_model(model.state_dict(), filename=(
            max_depth)+'_model_'+str(epoch)+'.pth.tar')
        print('model_'+str(epoch)+'.pth.tar DONE')

    return model


#nastavení učení a vytvoření datasetu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
learning_rate = 0.001
batch_size = 32
num_epoch = 50


#barevná augmentace pro trénovací dataset
trans = transforms.Compose([
    transforms.ColorJitter(brightness=0.4, contrast=0.4,
                           saturation=0.2, hue=0.2),
    transforms.ToTensor(),
])

#vytvoření datasetu pro trénování
train_dataset = AthleticDataset(
    pkl_file='new_train_data.pkl.gz', transform=trans)
test_dataset = AthleticDataset(
    pkl_file='new_test_data.pkl.gz', transform=transforms.ToTensor())

#loadery pro neuronové sítě
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset,
                         batch_size=batch_size, shuffle=True)

if __name__ == '__main__':
    #seznam možný modelů ze souboru models.py
    model_dict = {}
    #model_dict['classic_augment'] = CNN().to(device)
    #model_dict['classic_augment_dropoutV1'] = CNN_dropoutV1().to(device)
    #model_dict['classic_augment_dropoutV2'] = CNN_dropoutV2().to(device)
    model_dict['classic_augment_dropoutV3'] = CNN_dropoutV3().to(device)
    #model_dict['classic_augment_dropoutV4'] = CNN_dropoutV4().to(device)
    #model_dict['classic_augment_dropoutV5'] = CNN_dropoutV5().to(device)
    #model_dict['classic_augment_dropoutV6'] = CNN_dropoutV6().to(device)
    #model_dict['VGG_augment'] = VGGNetSeq().to(device)
    #model_dict['VGG_augment_dropout'] = VGGNetSeq_dropout().to(device)
    #model_dict['strip_augment'] = CNN_strip().to(device)
    #model_dict['strip_augment_dropout'] = CNN_strip_dropout().to(device)

    #připravený seznam pro jednotlivé architektury pro zaznamenání přesnosti
    acc_dict = {}
    #acc_dict['classic_augment'] = {'train': [], 'test': []}
    #acc_dict['classic_augment_dropoutV1'] = {'train': [], 'test': []}
    #acc_dict['classic_augment_dropoutV2'] = {'train': [], 'test': []}
    acc_dict['classic_augment_dropoutV3'] = {'train': [], 'test': []}
    #acc_dict['classic_augment_dropoutV4'] = {'train': [], 'test': []}
    #acc_dict['classic_augment_dropoutV5'] = {'train': [], 'test': []}
    #acc_dict['classic_augment_dropoutV6'] = {'train': [], 'test': []}
    #acc_dict['VGG_augment'] = {'train': [], 'test': []}
    #acc_dict['VGG_augment_dropout'] = {'train': [], 'test': []}
    #acc_dict['strip_augment'] = {'train': [], 'test': []}
    #acc_dict['strip_augment_dropout'] = {'train': [], 'test': []}

    for model_name in model_dict:
        print('___________________________________' +
              model_name+'___________________________________')
        model_dict[model] = train_model(
            model_dict[model_name], train_loader, test_loader, acc_dict, learning_rate, num_epoch, model_name)
