#!/usr/bin/env python3
# coding=utf-8

## Zpracování záznamu cílové kamery
# @file prepare_dataset.py
# @brief Příprava datasetu výřezů pro CNN
# @author Jahoda Vojtěch

__author__ = "Jahoda Vojtěch"
__description__ = "Příprava datasetu výřezů pro CNN"

import PIL
from PIL import Image
import numpy as np
import os
import json
import sys
import pandas as pd
import cv2
import gzip
import pickle

def name_label(f_path:str):
    """
    funkce pro předělání výřezu a anotace do tuple a připravení výřezu pro CNN
    @param f_path   -> cesta k výřezu
    @return         -> tuple(výřez pomocí knihovny PIL,anotace v rozsahu <0,1>)
    """
    path,filename = os.path.split(f_path)
    name,suffix = filename.split('.')
    suffx = suffix
    with open(f_path.replace(suffix, 'json'), "r") as f:
        js = json.loads(f.read())
    if type(js) == list:
        raise RuntimeError('výřez '+f_path+' nemá přesně určenou anotaci. v anotacich jsou '+str(js))
    return (PIL.Image.fromarray(cv2.resize(cv2.imread(f_path), (160, 160), interpolation=cv2.INTER_AREA)), js/190)

if __name__ == "__main__":
    cutout_path = 'cutouts'
    cutouts = [x for x in os.walk(cutout_path)]
    path_arr = []
    for folder in cutouts:
        for f in folder[2]:
            if 'json' not in f and 'eval' not in f:
                path_arr.append(name_label(os.path.join(folder[0], f)))

    with gzip.open('dataset.pkl.gz', 'w+') as f:
        pickle.dump(path_arr, f)
    

