#!/usr/bin/env python3
# coding=utf-8

## Zpracování záznamu cílové kamery
# @file CNN.py
# @brief vyhodnocení výřezu pomocí neuronové sítě
# @author Jahoda Vojtěch

__author__ = "Jahoda Vojtěch"
__description__ = "vyhodnocení výřezu pomocí neuronové sítě"

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as TModels

from PIL import Image
import os
import pandas as pd
import cv2
import numpy as np

import argparse

class CNN_dropoutV3(nn.Module):
    """
    vybraná architektura pro vyhodnocování cílových záznamů
    """
    def __init__(self, in_channels: int = 3, num_classes: int = 1):
        super(CNN_dropoutV3, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=8, kernel_size=(
            3, 3), stride=(1, 1), padding=(1, 1))
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(
            3, 3), stride=(1, 1), padding=(1, 1))
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(
            3, 3), stride=(1, 1), padding=(1, 1))
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(
            3, 3), stride=(1, 1), padding=(1, 1))

        self.fc1 = nn.Linear(6400, 2048)
        self.fc2 = nn.Linear(2048, num_classes)

        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.25)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.dropout1(x)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)

        return x


def load_model(model: CNN_dropoutV3, filename: str = 'model.pth.tar', folder: str = '') -> CNN_dropoutV3:
    """
    Funkce pro načtení předtrénovaných modelů
    @param model        -> objekt modelu který se má načíst
    @param filename     -> jméno uloženého modelu
    @param folder       -> složka, ve které je model uložen
    @return             -> načtzený předtrénovaný model
    """
    model.load_state_dict(torch.load(os.path.join(folder,filename)))
    return model

def image_loader(img_raw: np.ndarray) -> torch.tensor:
    """
    Funkce pro načtení jednotlivého výřezu jako tensoru
    @param img_raw  -> načítaný výřez
    @return         -> výřez v podobě tensoru
    """
    loader = transforms.Compose(
        [transforms.ToTensor()])
    img = Image.fromarray(cv2.resize(img_raw, (160, 160), interpolation=cv2.INTER_AREA))
    img = loader(img).float()
    img = img.unsqueeze(0)
    
    return img.to(device=device)

def eval_img(img_path: str , show: bool = False,reverse: bool = False,save: str = None) -> int:
    """
    Funkce pro vyhodnocení jednotlivých výřezů
    @param img_path     -> cesta k výřezu pro vyhodnocení
    @param show         -> přepínmač zobrazení výřezů
    @param reverse      -> přepínač pro obrácený doběh
    @return             -> posunutí (výsledek) na výřezu závodníka
    """
    print('[CNN] ',img_path)
    model.eval()
    with torch.no_grad():
        img_raw = cv2.imread(img_path)
        img = image_loader(img_raw)
        score = model(img)
        if show:
            img_eval = img_raw.copy()
            img_eval[:, int(score[0]*190)] = [0, 0, 255]
            Image.fromarray(cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)).show()
            Image.fromarray(cv2.cvtColor(img_eval, cv2.COLOR_BGR2RGB)).show()

        if not save is None:
            img_eval = img_raw.copy()
            img_eval[:, int(score[0]*190)] = [0, 0, 255]
            cv2.imwrite(save,img_eval)

        #posun v pixelech se určí pomocí vynásobení s původním rozměrem obrázku
        if reverse == True:
            #pro obrácený doběh je zapotžebí vracený posun zrcadlit
            return int((1-score[0])*img_raw.shape[0])
        else:
            return int(score[0]*img_raw.shape[0])


#globální proměnné
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNN_dropoutV3()
model = load_model(model).to(device)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Vyhodnocení výřezu závodníka za pomocí konvolučšní neuronové sítě', usage='%(prog)s --in_file [SOUBOR] -s')

    parser.add_argument('--in_file', metavar='Soubor',
                        help='Výřez závodníka ve formátu obrázku (JPG,PNG,JPEG...)')

    parser.add_argument(
        '--show', '-s', dest='show', action='store_true', default=False, help='zobrazení výřezů před i po vyhodnocení')

    parser.add_argument('--save_file', metavar='Soubor',
                        help='soubor, do kterého se uloží výsledný vyhodnocený výřez',default=None)

    args = parser.parse_args()

    if not os.path.exists(args.in_file):
        print('[problem] tento soubor neexistuje - ', args.in_file)


    eval_img(args.in_file,show=args.show,save=args.save_file)
    
    #print(torch.logical_and(scores < y+0.011,scores > y-0.011))
    
