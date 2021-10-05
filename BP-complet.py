#!/usr/bin/env python3
# coding=utf-8

## Zpracování záznamu cílové kamery
# @file BP-complet.py
# @brief Hlavní script pro zpracování cílových záznamů
# @author Jahoda Vojtěch

__author__ = "Jahoda Vojtěch"
__description__ = "Hlavní script pro zpracování cílových záznamů"

import cv2
import OpenPose_process as opp
import cutout
import os
import CNN
import argparse
from datetime import datetime
import shutil
from PIL import Image
import numpy as np

def eval_by_cnn(files: list, def_folder: str='tmp/cutouts', reverse: bool = False) -> list:
    """
    Funkce na získání jednotlivých vyhodnocení ze seznamu výřezů
    @param files        -> seznam tuple, ve kterých je posunutí celého výřezu na horizontální ose (base_offset) a jméno souboru výřezu
    @param def_folder   -> složka ve které jsou výřezy uloženy
    @param reverse      -> parametr pro obrácený výřez
    @return             -> seznam posunutí (výsledků) vůči hlavnímu obrázku
    """
    offsets = []
    for base_offset, f in files:
        offsets.append(CNN.eval_img(os.path.join(
            def_folder, f), show=False, reverse=reverse) + base_offset)
    return offsets


def complete_img(offsets: list, filename: str, show: bool = False, out_folder: str = 'out_photo') -> np.ndarray:
    """
    Funkce na skompletování vyhodnocených výřezů do původní celé ftografie
    @param offsets      -> seznam posunů (výsledků) pro jednotlivé závodníky
    @param filename     -> jméno původního souboru
    @param show         -> zobrazení výsledku pomocí knihovny PIL
    @param out_folder   -> složka pro výstupní fotografii
    @return             -> výsledná fotografie
    """
    img = cv2.imread(filename)
    for offset in offsets:
        img[:, offset] = [0, 0, 255]

    cv2.imwrite(os.path.join(out_folder, os.path.split(filename)[1]), img)

    if show:
        Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).show()
    
    return img


def complet_process_img(filename: str, out_folder: str = 'out_photo', reverse: bool = False, show: bool = False, rm_tmp: bool = True) -> np.ndarray:
    """
    Funkce pro provedení celého zpracování cílového záznamu
    @param filename     -> jméno vstupního souboru
    @param out_folder   -> složka pro uložení výstupu
    @param reverse      -> flag pro doběhnutí závodníka v opačném směru
    @param rm_tmp       -> flag pro odstanění dočasné složky tmp
    @return             -> výsledná fotografie
    """

    #vytváření potřebných složek
    if not os.path.exists('tmp'):
        os.mkdir('tmp')
    if not os.path.exists('tmp/json'):
        os.mkdir('tmp/json')
    if not os.path.exists('tmp/cutouts'):
        os.mkdir('tmp/cutouts')
    if not os.path.exists(out_folder):
        os.mkdir(out_folder)

    #OpenPose zpracování
    json_out = opp.process_image(filename.strip('.')[0], cv2.imread(filename))
    if type(json_out) != str:
        raise RuntimeError('OpenPose')
    print('Parsování knihovnou OpenPose se provedlo správně...')

    #vytvoření výřezů
    files = cutout.make_cutouts(filename, json_file=json_out, reverse=reverse)
    if type(files) != list:
        raise RuntimeError('Cutout')
    print('Vytváření výřezů proběhlo úspěšně.')

    #Vyhodnocení výřezlů pomocí neuronové sítě
    eval_offsets = eval_by_cnn(files, reverse=reverse)
    print('Vyhodnocení neuronovou sítí proběhlo úspěšně.')

    #kompletování fotografie
    completed_img = complete_img(eval_offsets, filename, show=show, out_folder=out_folder)

    if rm_tmp == True:
        shutil.rmtree('tmp')

    return complete_img


if __name__ == '__main__':
    start_time = datetime.now()

    parser = argparse.ArgumentParser(
        description='Vyhodnoceni cílových záznamů', usage='%(prog)s [možnosti]')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--in_file', metavar='Soubor',
                       help='Cílový záznam ve formátu obrázku (JPG,PNG...)')
    group.add_argument('--in_folder', metavar='Složka',
                       help='Složka s cílovýmy záýznamy pro vyhodnocení', default='in_photo')

    parser.add_argument('--out_folder', metavar='Složka',
                        help='Složka pro uložení cílových výsledků.', default='out_photo')

    parser.add_argument(
        '--reversed', '-r', dest='rev', action='store_true', default=False, help='flag pro záznamy, které jsou zaznamenané ve směru zprava doleva')

    parser.add_argument(
        '--show', '-s', dest='show', action='store_true', default=False, help='okamžité zobrazení výsledků po vyhodnocení')
    parser.add_argument(
        '--no_remove_tmp', dest='remove', action='store_false', default=True, help='Zachování obsahu složky tmp')

    args = parser.parse_args()
    if args.in_file is None:
        if not os.path.exists(args.in_folder):
            print('[problem] tato složka neexistuje - ', args.in_folder)
            raise FileNotFoundError

        dirs = [f for f in os.walk(args.in_folder)]
        for d in dirs:
            for f in d[2]:
                try:
                    complet_process_img(os.path.join(
                        d[0], f), out_folder=args.out_folder, reverse=args.rev, show=args.show, rm_tmp=args.remove)
                except RuntimeError as e:
                    print('error', e)
    else:
        if not os.path.exists(args.in_file):
            print('[problem] tento soubor neexistuje - ', args.in_file)
            raise FileNotFoundError

        filename = args.in_file
        try:
            complet_process_img(filename, out_folder=args.out_folder,
                                reverse=args.rev, show=args.show, rm_tmp=args.remove)
        except RuntimeError as e:
            print('error', e)

    print('doba vyhodnocení: ', (datetime.now() - start_time))
