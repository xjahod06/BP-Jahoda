# Automatické vyhodnocování cílových záznamů v atletice

## Prostředí
Primárně dělané pro Windows, ale mělo by fungovat i na Linux/OsX (netestováno)

## Autor
- Vojtěch Jahoda (xjahod06)

## Použití
V této složce jsou nástroje pro vytvoření datové sady výřezů z cílových záznamů. Postup je následovný:
Prvně je potřeba pomocí scriptu  `cutouts_dataset.py` vytvořit jednotlivé výřezy do složky `\cutouts`. Po vytvoření výřetů je zapotřebí zkontrolovat výřezy s malou jistotou za pomoci scriptu `review_dataset.py`, který postupně projde jednotlivé výřezy, které je potřeba ručně analyzovat a doupravit. Po zkontrolování všech výřezů stačí pouze vytvořir datovou sadu za pomocí scriptu `prepare_dataset.py`, který výřezy upraví do správné velikosti, připojí k nim anotaci a uloží do souboru dataset.pkl.tar.  Tento soubor následně stačí načíct do neuronové sítě.

příklad:
```
.\cutouts_dataset.py --in_file .\example\1.png --in_file_eval .\example\eval\1.png --in_json .\example\json\1.json
.\review_dataset.py
.\prepare_dataset.py
```

## seznam, souborů a složek v adresáři
• cutouts_dataset.py - soubor na vytvoření jednotlivých výřezů
• review_dataset.py - soubor pro kontrolu s menší jistotou
• prepare_dataset.py - soubor pro přípravu datasetu pro využití do CNN
• camera_photos.zip - datová sada celých cílových záznamů vyhodnocených OpenPose (CPU)
• \example - složka s jedním souborem pro demonstraci použití