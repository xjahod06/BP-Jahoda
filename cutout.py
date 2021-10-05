#!/usr/bin/env python3
# coding=utf-8

## Zpracování záznamu cílové kamery
# @file cutout.py
# @brief vytvoření výřezů pro závodníky
# @author Jahoda Vojtěch

__author__ = "Jahoda Vojtěch"
__description__ = "vytvoření výřezů pro závodníky"

import numpy as np
import os
import json
import cv2
import sys
import argparse
from PIL import Image
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


def save_img(img_data: np.ndarray, path: str, name: str) -> None:
    """
    Funkce pro uložení jednotlivých výřezů
    @param img_data -> vstupní výřez
    @param path     -> složka pro uložení
    @param name     -> jméno souboru pro uložení
    """
    print('[cutout] saving '+os.path.join(path, name))
    cv2.imwrite(os.path.join(path, name),
                cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR))


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

    """
    #ošetření proti špatnému náklonu kamery pro nejvzálenější dráhu
    if y < 90:
        y = 90
    data = img.copy()
    if rev:
        for i, row in enumerate(data[y-90:y+100, x-120:x+70]):
            #zjištění medianu (prahu) pro daný řádek
            threshold = matrix[y-90+i]
            for j, cell in enumerate(row):
                #porovnání pixelů vůči medianu na daný řádek
                if pixel_equal_ratio(threshold, cell, coef):
                    data[i+y-90, j+x-120, :] = 255
    else:
        for i, row in enumerate(data[y-90:y+100, x-70:x+120]):
            #zjištění medianu (prahu) pro daný řádek
            threshold = matrix[y-90+i]
            for j, cell in enumerate(row):
                #porovnání pixelů vůči medianu na daný řádek
                if pixel_equal_ratio(threshold, cell, coef):
                    data[i+y-90, j+x-70, :] = 255
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
    @return         -> výřez závodníka
    """
    if y < 90:
        y = 90
    if not points is None:
        full_img = parts(img, points)
    else:
        full_img = img.copy()
    if matrix is not None:
        full_img = whiten_cutout(full_img, y, x, matrix, rev=rev)
    if rev:
        return (x-120,mirror_img(full_img[y-90:y+100, x-120:x+70]))
    else:
        return (x-70, full_img[y-90:y+100, x-70:x+120])


def make_cutouts(filename: str, dir_out: str = 'tmp/cutouts', show: bool = False, save: bool = True, reverse: bool = False, json_file: str = None) -> list:
    """
    Funkce pro rozdělení cílové fotografie na jednotlivé výřezy
    @param filename     -> jméno cílového záznamu
    @param dir_out      -> složky pro výstupní výřezy
    @param show         -> ukázání jednotlivých výřezů
    @param save         -> přepínač ukládání výřezů
    @param reverse      -> flag pro cílový záznam pořízený v opačném směru
    @param json_file    -> soubor JSON, kerý obsahuje body pro jednbotlivé běžce z knihovny OpenPose
    @return             -> seznam jmen souborů výřezů s jejich posuny na záznamu
    """
    name,suffix = os.path.split(filename)[1].split('.')
    img = cv2.imread(filename)
    #změna barevného modelu, cv2 opoužívá defaultně BGR 
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    white_matrix = make_white_matrix(img)

    saved_files = []

    if json_file is None:
        raise FileExistsError
    else:
        with open(json_file, "r") as f:
            js = json.loads(f.read())


    for j, points in enumerate(js):
        if points[1][1] == 0 or points[1][0] == 0:
            continue
        x = int(points[1][0])
        y = int(points[1][1])
        base_offset, cut = make_cutout(img, y, x, points, white_matrix, reverse)

        if show:
            Image.fromarray(cut)

        if save:
            save_img(cut, dir_out, name+'_'+str(j)+'_('+str(base_offset)+').'+suffix)
            saved_files.append(
                (base_offset, name+'_'+str(j)+'_('+str(base_offset)+').'+suffix))

    return saved_files

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Vyhodnocení výřezu závodníka za pomocí konvolučšní neuronové sítě', usage='%(prog)s [možnosti]')

    parser.add_argument('--in_file', metavar='Soubor',
                        help='Výřez závodníka ve formátu obrázku (JPG,PNG,JPEG...)')
    
    parser.add_argument('--in_json', metavar='Soubor',
                        help='body závodníka z knihovny OpenPose ve formátu JSON')

    parser.add_argument('--out_folder', metavar='Složka',
                        help='složka pro uložení výsledných výřezů')

    parser.add_argument(
        '--show', '-s', dest='show', action='store_true', default=False, help='zobrazení výřezů před i po vyhodnocení')

    parser.add_argument(
        '--no_save', dest='save', action='store_false', default=True, help='uložení výřezů do složky')
    parser.add_argument(
        '--reversed', '-r', dest='rev', action='store_true', default=False, help='flag pro záznamy, které jsou zaznamenané ve směru zprava doleva')

    args = parser.parse_args()

    if not os.path.exists(args.in_file):
        print('[problem] tento soubor neexistuje - ', args.in_file)

    make_cutouts(filename=args.in_file,dir_out=args.out_folder,show=args.show,save=args.save,reverse=args.rev,json_file=args.in_json)
    
    

