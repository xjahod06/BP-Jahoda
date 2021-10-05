#!/usr/bin/env python3
# coding=utf-8

## Zpracování záznamu cílové kamery
# @file cutouts_dataset.py
# @brief vytvoření výřezů do datasetu
# @author Jahoda Vojtěch

__author__ = "Jahoda Vojtěch"
__description__ = "vytvoření výřezů do datasetu"

import numpy as np
import os
import json
import cv2
import sys
import argparse
from PIL import Image

import shutil
from datetime import datetime
import stat
import random

np.seterr(divide='ignore', invalid='ignore')


def place_point(img: np.ndarray, x: float, y: float, score: float, radius: int = 3, color: list = [255, 0, 0]) -> np.ndarray:
    """
    Funkce pro vykreslení body na výřez
    @param img      -> vstupní obrázek
    @param x        -> souřadnice x pokládaného bodu
    @param y        -> souřadnice y pokládaného bodu
    @param score    -> jistota daného bodu
    @param radius   -> poloměr vykresleného bodu
    @param color    -> barva bodu podle barevného modelu RGB
    @return         -> obrázek s vykresleným bodem
    """
    x = int(x)
    y = int(y)
    img[x-radius:x+radius, y-radius:y +
        radius] = np.array([color[0]*score, color[1], color[2]])
    return img


def mirror_img(img: np.ndarray) -> np.ndarray:
    """
    Funkce pro zrcadlení obrázku (otočení kolem vertikální osy)
    @param img  -> vstupní obrázek
    @return     -> výstupní zrcadlený obrázek
    """
    if type(img) == tuple:
        return img[0][:, ::-1, :], img[1]
    else:
        return img[:, ::-1, :]


def parts(data: np.ndarray, points: np.ndarray) -> np.ndarray:
    """
    Funkce pro určen, které body se mají zakreslit do výřezu
    @param data     -> vstupní obrázek
    @param points   -> seznam bodů postavy z OpenPose
    @return         -> výřez s zakreslenými body

    @note následující body jsou určřeny podle modelui BODY_25 knihovnou OpenPose
    """
    data_points = data.copy()
    data_points = place_point(
        data_points, points[0][1], points[0][0], 1, color=[123, 0, 123])  # head
    data_points = place_point(
        data_points, points[1][1], points[1][0], 1)  # body
    data_points = place_point(data_points, points[2][1], points[2][0], 1, color=[
                              0, 255, 0])  # left sholder
    data_points = place_point(data_points, points[5][1], points[5][0], 1, color=[
                              0, 0, 255])  # right sholder
    return data_points


def get_offset(offsets: list, competitor: np.ndarray, rev:bool = False):
    """
    Funkce pro získání možných anotací pro běžce
    @param offsets -> seznam posunů (anotací) pro celý cílový záznam
    @param competitor -> seznam bodů pro daného běžce
    @param rev -> přepínaš pro otočení směro doběhu závodníka
    @return -> tuple(None,None) v případě, kdy nebyla nalezena žádná potencionální anotace
               tuple(offset,None) v případě, kdy byla nalezena pouze jedna anotace
               tuple(None,List) v případě, kdy bylo nalezeno více anotací
    """
    x = int(competitor[1][0])

    #získání počtu anotací, které zapadají so rozsahu výřezu
    if rev:
        result = np.where(np.logical_and(offsets >= x-120, offsets <= x+70))
    else:
        result = np.where(np.logical_and(offsets >= x-70, offsets <= x+120))

    potencional_offsets = np.array([offsets[x] for x in result[0]])

    if len(potencional_offsets) == 0:
        return (None, None)
    elif len(potencional_offsets) == 1:
        return (potencional_offsets[-1], None)
    else:
        return (None, potencional_offsets)




def save_img(img_data: np.ndarray, flag: str, path: str, name: str, img_with_res: np.ndarray, result: 'anotace'):
    """
    Funkce pro uložení jednotlivých výřezů
    @param img_data         -> vstupní výřez
    @param flag             -> flag určující přesnost s jakou byl daný výřez pořízen
    @param path             -> složka pro uložení
    @param name             -> jméno souboru pro uložení
    @param img_with_res     -> Uložení i již vyhodnocené varianty pro pozdější vyhodnocení přesné anotace (více anotací)
    @param result           -> anotace pro jednotlivý výřez
    """
    if flag == 'valid':
        path = os.path.join(path, 'correct')
    elif flag == 'low_acc':
        path = os.path.join(path, 'to_review')
    else:
        pass
    
    name,suffix = name.split('.')

    suffix = '.'+suffix

    cv2.imwrite(os.path.join(path, name+suffix),
                cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR))
    with open(os.path.join(path, name+'.json'), 'w+') as f:
        json.dump(result.tolist(), f, separators=(',', ':'))

    if not img_with_res is None:
        cv2.imwrite(os.path.join(path, name+'_eval'+suffix),
                    cv2.cvtColor(img_with_res, cv2.COLOR_RGB2BGR))


def aug_resize(img: np.ndarray, x: list, y: list) -> np.ndarray:
    """
    Funkce pro provedení augmentace zvětšením/zmenšením výřezu
    @param img  -> vstupní výřez
    @param x    -> souřadnice výřezu vůči celému záznamu na ose x
    @param y    -> souřadnice výřezu vůči celému záznamu na ose y
    @return     -> augmentovaný výřez, 0 (posun kvůli anotaci)
    """
    shift = random.randint(0, 3)
    if random.randint(0, 1) == 0:
        augmented = img[y[0]+shift:y[1]-shift, x[0]+shift:x[1]-shift]
        return cv2.resize(augmented, (x[1]-x[0], y[1]-y[0]), interpolation=cv2.INTER_AREA), 0
    else:
        augmented = cv2.resize(
            img[y[0]:y[1], x[0]:x[1]], (y[1]-y[0]+shift*2, x[1]-x[0]+shift*2), interpolation=cv2.INTER_AREA)
        return augmented[1:-1, 1:-1], 0


def aug_move(img: np.ndarray, x: list, y: list) -> np.ndarray:
    """
    Funkce pro provedení augmentace posunu výřezu
    @param img  -> vstupní výřez
    @param x    -> souřadnice výřezu vůči celému záznamu na ose x
    @param y    -> souřadnice výřezu vůči celému záznamu na ose y
    @return     -> augmentovaný výřez, posun (kvůli anotaci)
    """
    shift = random.randint(0, 4)
    switch = random.randint(0, 7)
    if switch == 0:
        return img[y[0]-shift:y[1]-shift, x[0]:x[1]], 0
    elif switch == 1:
        return img[y[0]+shift:y[1]+shift, x[0]:x[1]], 0
    elif switch == 2:
        return img[y[0]:y[1], x[0]+shift:x[1]+shift], -shift*2
    elif switch == 3:
        return img[y[0]:y[1], x[0]-shift:x[1]-shift], shift*2
    elif switch == 4:
        return img[y[0]+shift:y[1]+shift, x[0]+shift:x[1]+shift], -shift*2
    elif switch == 5:
        return img[y[0]+shift:y[1]+shift, x[0]-shift:x[1]-shift], shift*2
    elif switch == 6:
        return img[y[0]-shift:y[1]-shift, x[0]+shift:x[1]+shift], -shift*2
    elif switch == 7:
        return img[y[0]-shift:y[1]-shift, x[0]-shift:x[1]-shift], shift*2


def augment(img: np.ndarray, x: list, y: list) -> np.ndarray:
    """
    Funkce pro výběr augmentace výřezu
    @param img  -> vstupní výřez
    @param x    -> souřadnice výřezu vůči celému záznamu na ose x
    @param y    -> souřadnice výřezu vůči celému záznamu na ose y
    @return     -> augmentovaný výřez, posun (kvůli anotaci)
    """
    aug_list = {
        0: aug_resize,
        1: aug_move,
        2: aug_move
    }
    img = aug_list[random.randint(0, 2)](img, x, y)
    return img


def make_white_matrix(img: np.ndarray) -> np.ndarray:
    """
    Funkce pro vytvoření "matice" medianu na řádku
    @param img  -> vstupní obrázek
    @return     -> matice medianu 
    """
    matrix = np.empty([img.shape[0], 3])
    for i, line in enumerate(img):
        matrix[i] = np.array([np.median(img[i, :, 0]), np.median(
            img[i, :, 1]), np.median(img[i, :, 2])])
    return matrix


def isclose(a: float, b: float, tolerance: float) -> bool:
    """
    Funkce pro porovnání podobnosti čísel
    @param a            -> číslo a
    @param b            -> číslo b
    @param tolerance    -> povolený rozdíl mezi těmito čísly
    @return             -> pravdivostní hodnota zda si jsou čísla podobná či nikoliv
    """
    return abs(a-b) <= tolerance


def pixel_equal_ratio(new: list, old: list, deviation_per: float) -> bool:
    """
    Funkce pro porovnání pixelů na základě poměru barev mez ijednotlivými barevnými složkami
    @param new              -> nový pixel na porovnání
    @param old              -> starý pixel na porovnání 
    @param deviation_per    -> koeficient na porovnání pixelu
    @return                 -> pravivostní hodnota o podobnosti pixelu
    """
    deviation = deviation_per*255
    sm = old[:].sum()
    if sm == 0:
        sm = sys.float_info.epsilon
    #vytvoření poměrů jednmotlivých barevných složek
    ratios = old[:] / old[:].sum()
    npa = np.array([0, 0, 0])
    for i in range(3):
        npa[i] = isclose(old[i], new[i], deviation*ratios[i])
    return(npa.all())


def whiten_cutout(img: np.ndarray, y: float, x: float, matrix: np.ndarray, coef: float = 0.1, rev: bool = False) -> np.ndarray:
    """
    Funkce na "vybělení" výřezů pomocí segmentace pozadí
    @param img      -> vstupní výřez
    @param y        -> pozice hrudníku závodníka na ose y
    @param x        -> pozice hrudníku závodníka na ose x
    @param matrix   -> matice medinu pro každý jednotlivý řádek
    @param coef     -> koeficient pro segmentaci pozadí
    @param rev      -> flag pro otočení závodníka
    @return         -> výřez s segmentovaným pozadím

    @note zde je zapotřebí "vybělit" o něco větší čtverec kolem závodníka kvůli případné augmentaci posunu
    """
    #ošetření proti špatnému náklonu kamery pro nejvzálenější dráhu
    if y < 94:
        y = 94
    data = img.copy()
    if rev:
        for i, row in enumerate(data[y-94:y+104, x-124:x+74]):
            #zjištění medianu (prahu) pro daný řádek
            threshold = matrix[y-94+i]
            for j, cell in enumerate(row):
                #porovnání pixelů vůči medianu na daný řádek
                if pixel_equal_ratio(threshold, cell, coef):
                    data[i+y-94, j+x-124, :] = 255
    else:
        for i, row in enumerate(data[y-94:y+104, x-74:x+124]):
            #zjištění medianu (prahu) pro daný řádek
            threshold = matrix[y-94+i]
            for j, cell in enumerate(row):
                #porovnání pixelů vůči medianu na daný řádek
                if pixel_equal_ratio(threshold, cell, coef):
                    data[i+y-94, j+x-74, :] = 255
    return data


def make_cutout(img: np.ndarray, y: float, x: float, points: np.ndarray, matrix: np.ndarray, rev: bool) -> np.ndarray:
    """
    Funkce pro vytvoření samotného výřezu
    @param img      -> vstupní obrázek
    @param y        -> pozice hrudníku závodníka na ose y
    @param x        -> pozice hrudníku závodníka na ose x
    @param points   -> seznam bodů částí těla z knihovny OpenPose
    @param matrix   -> matice medianu pro každý řádek (pro segmentace pozadí)
    @param rev      -> flag pro otočení fotografie
    @return         -> augmentaci výřezu závodníka, originální výřez
    """
    if y < 94:
        y = 94
    if not points is None:
        full_img = parts(img, points)
    else:
        full_img = img.copy()
    if matrix is not None:
        full_img = whiten_cutout(full_img, y, x, matrix, rev=rev)
    
    if rev:
        return mirror_img(augment(full_img,(x-120,x+70),(y-90,y+100))),mirror_img(full_img[y-90:y+100,x-120:x+70])
    else:
        return augment(full_img,(x-70,x+120),(y-90,y+100)),full_img[y-90:y+100,x-70:x+120]

def check_dirs():
    """
    Funkce pro přípravu adresářové struktury na vytvoření datové sady
    """
    dirs = ['cutouts',
            'cutouts/single',
            'cutouts/single/correct',
            'cutouts/single/to_review',
            'cutouts/multi',
            'cutouts/multi/correct',
            'cutouts/multi/to_review',
            'cutouts/augmented']
    for directory in dirs:
        if not os.path.exists(directory):
            os.mkdir(directory)

def make_cutouts(filename: str, eval_file: str, dir_out: str = 'cutouts', show: bool = False, save: bool = True, reverse: bool = False, json_file: str = None) -> list:
    """
    Funkce pro rozdělení cílové fotografie na jednotlivé výřezy
    @param filename     -> jméno cílového záznamu
    @param eval_file    -> již vyhodnocený cílový záznam 
    @param dir_out      -> složky pro výstupní výřezy
    @param show         -> ukázání jednotlivých výřezů
    @param save         -> přepínač ukládání výřezů
    @param reverse      -> flag pro cílový záznam pořízený v opačném směru
    @param json_file    -> soubor JSON, kerý obsahuje body pro jednbotlivé běžce z knihovny OpenPose
    @return             -> seznam jmen souborů výřezů s jejich posuny na záznamu
    """
    check_dirs()
    name, suffix = os.path.split(filename)[1].split('.')
    suffix = '.'+suffix
    img = cv2.imread(filename)
    #změna barevného modelu, cv2 opoužívá defaultně BGR
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img_results = cv2.imread(eval_file)
    img_results = cv2.cvtColor(img_results, cv2.COLOR_BGR2RGB)

    white_matrix = make_white_matrix(img)

    saved_files = []

    if json_file is None:
        raise FileExistsError
    else:
        with open(json_file, "r") as f:
            js = json.loads(f.read())

    js.sort(key=lambda competitor: competitor[1][0])

    #anotace
    offsets = []
    for i, point in enumerate(img_results[0]):
        # hledání horizontální pozice vyhodnocených čar
        if np.array_equal(point, np.array([220, 10, 10])):
            offsets.append(i)
    offsets = np.array(offsets)

    for j, points in enumerate(js):
        if points[1][1] == 0 or points[1][0] == 0:
            continue

        #nastavení flagu jistoty pdoel jistoty bodů z OpenPose
        flag = 'valid'
        if points[1][2] < 0.7:
            flag = 'low_acc'

        x = int(points[1][0])
        y = int(points[1][1])

        actual_offset, potentional_offsets = get_offset(
            offsets, points, reverse)

        #base_offset, cut = make_cutout(
        #    img, y, x, points, white_matrix, reverse)

        (aug, aug_offset), cut = make_cutout(
            img, y, x, points, white_matrix, reverse)

        _, cut_with_res = make_cutout(
            img_results, y, x, points, white_matrix, reverse)
        
        if show:
            Image.fromarray(cut)

        if actual_offset == None and potentional_offsets is None:
            print(
                '[Problem] Pro některého ze závodníků chybí anotace. souřadnice', x, y)
            continue
        
        #jedna anotace na jednoho závodníka
        elif actual_offset != None:

            #posun anotaci kvůli augmentacím
            if reverse:
                actual_offset = actual_offset - (x-120) + aug_offset
            else:
                actual_offset = actual_offset - (x-70) + aug_offset

            if save:
                save_img(cut, flag, os.path.join(dir_out, 'single'), name+'_'+str(j)+suffix, None, np.array(actual_offset))
                save_img(aug, 'augmented', os.path.join(
                    dir_out, 'augmented'), name+'_'+str(j)+suffix, None, np.array(actual_offset))
        
        #více anotací na jednoho závodníka
        elif actual_offset == None and not potentional_offsets is None:
            
            #posun anotaci kvůli augmentacím
            if reverse:
                potentional_offsets = potentional_offsets - (x-120) + aug_offset
            else:
                potentional_offsets = potentional_offsets - (x-70) + aug_offset
            #print(potentional_offsets)
            if save:
                save_img(cut,flag,os.path.join(dir_out,'multi'),name+'_'+str(j)+suffix,cut_with_res,potentional_offsets)
                save_img(aug,'augmented',os.path.join(dir_out,'augmented'),name+'_'+str(j)+suffix,None,potentional_offsets)


        #if save:
        #    save_img(cut, dir_out, name+'_'+str(j) +
        #             '_('+str(base_offset)+').'+suffix)
        #    saved_files.append(
        #        (base_offset, name+'_'+str(j)+'_('+str(base_offset)+').'+suffix))

    return saved_files


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Vyhodnocení výřezu závodníka za pomocí konvolučšní neuronové sítě', usage='%(prog)s [možnosti]')

    parser.add_argument('--in_file', metavar='Soubor',
                        help='Výřez závodníka ve formátu obrázku (JPG,PNG,JPEG...)')

    parser.add_argument('--in_file_eval', metavar='Soubor',
                        help='Výřez závodníka ve formátu obrázku (JPG,PNG,JPEG...)')

    parser.add_argument('--in_json', metavar='Soubor',
                        help='body závodníka z knihovny OpenPose ve formátu JSON')

    parser.add_argument(
        '--show', '-s', dest='show', action='store_true', default=False, help='zobrazení výřezů před i po vyhodnocení')

    parser.add_argument(
        '--no_save', dest='save', action='store_false', default=True, help='uložení výřezů do složky')
    parser.add_argument(
        '--reversed', '-r', dest='rev', action='store_true', default=False, help='flag pro záznamy, které jsou zaznamenané ve směru zprava doleva')

    args = parser.parse_args()

    if not os.path.exists(args.in_file):
        print('[problem] tento soubor neexistuje - ', args.in_file)

    make_cutouts(filename=args.in_file,
                 eval_file=args.in_file_eval,
                 show=args.show,
                 save=args.save,
                 reverse=args.rev,
                 json_file=args.in_json)
