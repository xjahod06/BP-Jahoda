# Automatické vyhodnocování cílových záznamů v atletice

## Prostředí
Primárně dělané pro Windows, ale mělo by fungovat i na Linux/OsX (netestováno)

## Autor
- Vojtěch Jahoda (xjahod06)

## Požadavky na spuštění
- python verze 3.7 (záleží na OpenPose)
- Knihovna OpenPose přeložena jako Python API ve složce 'OpenPose_lib'
- nástroj CUDA a případně cuDNN

Využívané knihovny pro Python
- numpy
- opencv (cv2)
- torch (verze podporující CUDA)
- torchvision (verze podporující CUDA)
- pandas
- PIL
- json
- gzip
- matplotlib

## Instalace OpenPose
V mém případě bylo zapotřebí prvně stáhnout celý github adresář z https://github.com/CMU-Perceptual-Computing-Lab/openpose a pomocí nástroje CMAKE vygenerovat projekt OpenPose jako python API. Po přeložení tohot projektu pomocíé visual studia jako `release` jsem byl schopen jej spustit. Jelikož nedisponuji vysokým výpočetním výkonem GPU, tak je tomto adresáři tato knihovna přeložena na verzi využívající CPU (procesor).
Tento postup se mi bohužel nepovedlo zopakovat na notebooku, takže tam existují nějaké proměnné pro každé zařízení a nejsem schopný tedy přesně určit univerzální návod pro každé zařízení. 
Postup oficiální instalace je vysvětlen v oficiální dokumentace k OpenPose na adrese https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/installation/0_index.md#compiling-and-running-openpose-from-source .

## Použití
Pro použití tohoto programu je zapotřebí mít cílový záznam ve formátu obrázku (JPG,PNG...). Program se dělí do několika scriptů, kdy hlavním je `BP-complet.py`.

Následné použití je pomocí příkazové řádky dle parametrů:
```
  -h, --help           zobrazení nápovědy
  --in_file Soubor     Cílový záznam ve formátu obrázku (JPG,PNG...)
  --in_folder Složka   Složka s cílovýmy záýznamy pro vyhodnocení
  --out_folder Složka  Složka pro uložení cílových výsledků.
  --reversed, -r       flag pro záznamy, které jsou zaznamenané ve směru
                       zprava doleva
  --show, -s           okamžité zobrazení výsledků po vyhodnocení
  --no_remove_tmp      Zachování obsahu složky tmp
```
kdy použití parametru `in_file` je vyhodnocení pouze jednoho souboru a nedá se kombinovat s parametrem `in_folder`, který určuje vyhodnocení všech cílových záznamů v dané složce.

příklad použití:
```
.\BP-complet.py --in_file in_photo\1.png -s
```
vygeneruje vyhodnocený záznam z cílového a zobrazí jej po dokončení.


```
.\BP-complet.py --in_file in_photo\15.png -s -r
```
vygeneruje vyhodnocený záznam z cílového a zobrazí jej po dokončení, flag -r protože cílový záznam 15.png je pořízen v opačném směru.

## seznam, souborů a složek v adresáři
• OpenPose_lib - Přiložená knihovna OpenPose pro python v „módu“ CPU.
• dataset - Náležitosti potřebné pro vytvoření datasetu.
• CNN_uceni - Složka obsahujici scripty pro trénování neuronové sítě.
• BP-complet.py - Hlavní soubor pro spuštění.
• OpenPose_process.py - Soubor pro zpracování obrázků knihovnou OpenPose.
• cutout.py - Soubor pro vytvoření výřezů závodníků.
• CNN.py - Soubor obsahující konvoluční neurunovou síť a její vyhodnocení.
• README.md - soubor obsahující informace pro zprovoznění a ovládání programu.
• model.pth.tar - předučený model konvoluční neuronové sítě.
