#!/usr/bin/env python3
# coding=utf-8

## Zpracování záznamu cílové kamery
# @file review_cutouts.py
# @brief Prostředek pro ruční určení anotací
# @author Jahoda Vojtěch

__author__ = "Jahoda Vojtěch"
__description__ = "Prostředek pro ruční určení anotací"

from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import os
import json
import shutil
from datetime import datetime
import cv2
import sys
import stat
import random


def move_to_scrap(f_path:str,eval:bool=False):
    """
    Funkce pro odstranění chybných souborů
    @param f_path -> soubor na odstanění
    @param eval   -> přepínač pokud byl vegenerován i již předvyhodnocený výřez
    """
    os.remove(f_path)
    if eval:
        name,suffix = f_path.split('.')
        suffix = '.'+suffix
        os.remove(f_path.replace(suffix,'_eval'+suffix))


def single_review(f_path:str):
    """
    funkce pro rozhodnutí zda výřez má patřit do datové sady či nikoliv
    """
    img = cv2.imread(f_path)

    #zobrazení výřezu
    cv2.imshow(f_path, img)
    cv2.waitKey(1)

    answer = input('Y/N?')
    cv2.destroyAllWindows()
    cv2.waitKey(1)


    if answer == 'N' or answer == 'n':
        move_to_scrap(f_path)
        move_to_scrap(f_path.replace('single\\to_review', 'augmented'))

def json_change(f_path:str, idx:int):
    """
    Funkce pro přepis json souboru (anotaci) z listu hodnotu podle indexu
    @param f_path -> cesta k souboru
    @param idx    -> vybraný index v listu
    """
    with open(f_path, "r") as f:
        js = json.loads(f.read())
    with open(f_path, 'w+') as f:
        json.dump(js[idx], f, separators=(',', ':'))


def multi_review(f_path):
    """
    Funkce na vyhodnocení 'multi' anotací u výřezů
    @param -> vstupní soubor
    """
    img_path = f_path.replace('_eval', '')
    img = cv2.imread(f_path)
    path,filename = os.path.split(f_path)
    name,suffix = filename.split('.')
    try:
        with open(img_path.replace(suffix, 'json'), "r") as f:
            js = json.loads(f.read())
    except FileNotFoundError:
        print('soubor Json pro',f_path,'nebyl nalezen')
        return
    except json.decoder.JSONDecodeError as e:
        js = []
        for i, point in enumerate(img[0]):
            if np.array_equal(point, np.array([10, 10, 220])):
                js.append(i)

    if type(js) == int:
        print(f_path,'už byl vyhodnocen')
        return
    cv2.imshow(f_path, img)
    cv2.waitKey(1)
    answer = input('Y/N?')
    cv2.destroyAllWindows()
    cv2.waitKey(1)

    if answer == 'N' or answer == 'n':
        move_to_scrap(img_path, eval='true')
        move_to_scrap(img_path.replace('multi', 'augmented').replace(
            '\\to_review', '').replace('\\correct', ''))
    else:
        json_change(img_path.replace(suffix,'json'), int(answer)-1)
        json_change(img_path.replace('multi', 'augmented').replace(
            '\\to_review', '').replace('\\correct', '').replace(suffix,'json'), int(answer)-1)

if __name__ == '__main__':
    dir_path = os.path.dirname(os.path.realpath(__file__))
    print('schvalování výřezů ze složky to_review, kdy pomocí písmen y a n se určuje zda daný výřez přežije či nikoliv:')
    cutout_path = 'cutouts'
    cutouts = [x for x in os.walk(cutout_path)]
    for folder in cutouts:
        if 'single' in folder[0] and 'to_review' in folder[0]:
            for f in folder[2]:
                if 'json' not in f:
                    single_review(os.path.join(dir_path, folder[0], f))

    input("jednotlivé výřezy jsou hotovy, dále následují 'multi'. Stisknutím pokračujte...")
    print("""schvalování výřezů ze složky to_review, kdy pomocí písmen y a n se určuje zda daný výřez přežije či nikoliv.
            Pro určení správné anotace výřezu stačí napsat číslo vybrané čáry na výřezu.
            Anotace jsou číslové zleva doprava a stačínají se počíatat od 1""")
    cutouts = [x for x in os.walk(cutout_path)]
    for folder in cutouts:
        if 'multi' in folder[0]:
            for f in folder[2]:
                if 'json' not in f and 'eval' in f:
                    multi_review(os.path.join(dir_path, folder[0], f))
        else:
            continue
    print('end')
