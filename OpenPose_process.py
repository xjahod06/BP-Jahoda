#!/usr/bin/env python3
# coding=utf-8

## Zpracování záznamu cílové kamery
# @file OpenPose_process.py
# @brief scripr pro zpracování záznamu cílovou kamerou
# @author Jahoda Vojtěch

__author__ = "Jahoda Vojtěch"
__description__ = "scripr pro zpracování záznamu cílovou kamerou"



import json
import sys
import cv2
import os
from sys import platform
import argparse
import numpy as np
import argparse


def process_image(name: str,img: np.ndarray, json_save: bool = True, json_path: str = 'tmp/json') -> str:
    """
    Funkce pro zpracování cílového záznamu knihovnou OpenPose
    @param name         -> jméno souboru cílového záznamu
    @param img          -> vstupní obrázek pro cílový soubor
    @param json_save    -> přepínač ukládání JSON souboru
    @param json_path    -> složka na uložení JSON souborů
    @return             -> jméno uloženého JSON souboru

    @note tato funkce je inspirována příklady použití knihovny OpenPose pro Python knihovnu
    """
    # Import knihovny OpenPose pro Windows\Linus\OsX
    dir_path = os.path.dirname(os.path.realpath(__file__))

    try:
        # Windows Import
        if platform == "win32":
            #Zde musí pro windows být nastavena relativní cesta od tohoto souboru pro složky Release a Bin
            #Pro přesun knihovny OpenPose je zapotřebý tyto proměnné přepsat
            def_folder = '/OpenPose_lib/build'
            sys.path.append(dir_path + def_folder + '/python/openpose/Release')
            os.environ['PATH'] = os.environ['PATH'] + ';' + dir_path + \
                def_folder + '/x64/Release;' + dir_path + def_folder + '/bin;'
            import pyopenpose as op
        else:
            #Zde musí pro windows být nastavena relativní cesta od tohoto souboru pro složku python v přeložené knihovně OpenPose
            sys.path.append('/OpenPose_lib/build/python')
            from openpose import pyopenpose as op
    except ImportError as e:
        # chyba importu, pokud knihovna OpenPose není přelkožena jako python knihovna
        print('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake?')
        raise e

    #nastavení relativní cesty k modelům OpenPose
    params = dict()
    params["model_folder"] = "OpenPose_lib/models/"

    # Starting OpenPose
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()

    # Process Image
    datum = op.Datum()
    datum.cvInputData = img
    opWrapper.emplaceAndPop(op.VectorDatum([datum]))

    # uložení bodů do JSON souboru
    if json_save:
        js_data = datum.poseKeypoints.tolist()
        with open(os.path.join(json_path,name+'.json'), 'w+') as f:
            json.dump(js_data, f, separators=(',', ':'))

    return os.path.join(json_path,name+'.json')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='získájní bodů postavy z knihovny OpenPose', usage='%(prog)s --in_file')

    parser.add_argument('--in_file', metavar='Soubor',
                        help='Cílový záznam ve formátu obrázku (JPG,PNG,JPEG...)')

    args = parser.parse_args()

    if not os.path.exists(args.in_file):
        print('[problem] tento soubor neexistuje - ', args.in_file)
        raise FileNotFoundError

    process_image(os.path.split(args.in_file)[1], cv2.imread(args.in_+file))
