# BIOMETRICKÉ SYSTÉMY
## Klasifikátor hodnocení kvality obrazu sítnice
### Tomáš Ondrušek, Peter Ďurica
#### FIT VUT v Brne, 2023

## Úvod

Tento projekt je zameraný na klasifikáciu kvality obrazu sítnice. Vstupom sú snímky sietnice, ktoré sú získané pomocou fundus kamery. V tomto projekte sú použité dve sady snímkov. Jedna je z datasetu *EyePACS*, dostupný na stránke [Kaggle](https://www.kaggle.com/c/diabetic-retinopathy-detection/data). Druhá je z datasetu *STRaDe*. Snímky v datasetoch sú klasifikované do 3 tried podľa stupňa kvality obrazu. Tieto triedy sú: 
-    *Good*
 -   *Usable*
  -  *Reject*
  
Ohodnotenia týchto datasetov sú v priečinku *data/* v súboroch *\*_labels.csv*.

## Popis riešenia

Riešenie je rozdelené na 3 časti. Prvá časť je zameraná na ohodnotenie dát, druhá na predspracovanie dát. Tretia časť je zameraná na samotnú klasifikáciu.

### Ohodnotenie dát

Ohodnotenie dát je implementované v priečinku *anotator_gui/*. Jedná sa o jednoduchú aplikáciu, ktorá umožňuje ohodnotiť snímky sietnice. Aplikácia je implementovaná pomocou knižnice *Tkinter*. Aplikácia je rozdelená na 3 časti. Aplikácia pri spustení buď načíta už existujúci csv súbor s ohodnoteniami, alebo vytvorí nový. Aplikácia následne zobrazí snímok sietnice. Po zobrazení snímku je možné snímok ohodnotiť pomocou tlačidiel *Good*, *Usable* a *Reject*. Po ohodnotení snímku sa automaticky zobrazí ďalší snímok. Po ohodnotení všetkých snímkov sa aplikácia ukončí. Tento súbor je následne možné použiť na trénovanie a testovanie modelov.

Ak nemáme vopred rozdelené snímky na trénovacie a testovacie, je možné použiť script *split.py* v priečinku *preprocess/*. Tento script rozdelí snímky na trénovacie a testovacie v pomere určenom parametrom *--threshold*. Tento script je možné spustiť pomocou príkazu:

```python split.py --input_path ___ --train_output_path ___ --test_output_path ___ --threshold n```

Kde *--input_path* je cesta k súboru s označením snímkov, *--train_output_path* je cesta k súboru, kde sa majú uložiť označenia trénovacích snímkov, *--test_output_path* je cesta k súboru, kde sa majú uložiť označenia testovacích snímkov a *--threshold* je pomer rozdelenia snímkov na trénovacie a testovacie.


### Predspracovanie dát

Predspracovanie dát je implementované v priečinku *preprocess/*. Princíp predspracovania je nasledovný:

1. Načítanie snímky
2. Detekcia okraja snímky
3. Výber oblasti záujmu
4. Získanie masky oblasti záujmu
5. Aplikácia masky na snímku
6. Orezanie snímky podľa masky do štvorca
7. Normalizácia snímky na veľkosť 800x800

Pre zlepšenie rýchlosti spracovania je predspracovanie implementované paralelne pomocou knižnice *dask*. Výsledky predspracovania sú uložené v priečinku *images/*.

### Klasifikácia

Na klasifikáciu sú použité 3 modely, a to *ResNet18*, *ResNet50* a *DenseNet121*. Tieto modely sú implementované v knižnici *PyTorch*. Tieto modely sú inštancované 3 krát, a to pre každú farebnú spektrálnu zložku snímky, a to RGB, HSV a LAB. Tieto modely sú následne spojené do jedného modelu, ktorý je použitý na klasifikáciu, ako je naznačené v článku [Evaluation of Retinal Image Quality Assessment Networks in Different Color-spaces](https://link.springer.com/chapter/10.1007/978-3-030-32239-7_6).

Princíp klasifikácie je nasledovný:

1. Načítanie snímky
2. Rozdelenie snímky do troch farebných spektier, a to RGB, HSV a LAB
3. Vytvorenie datasetu z týchto troch spektier
4. Uloženie datasetu do tensoru
5. Trénovanie modelu na datasete cez n epochov
6. Uloženie najlepšieho modelu
7. Testovanie modelu na testovacích dátach
8. Výpis presnosti modelu

## Spustenie

### Predspracovanie dát

Predspracovanie dát je možné spustiť pomocou príkazu:

```python preprocess.py  --save_path ../images/___ --path ../images/___.zip```

Kde *--save_path* je cesta, kde sa majú uložiť predspracované snímky a *--path* je cesta k zip súboru so snímkami.

### Klasifikácia

Ako nápoveda pre spustenie je možné použiť príkaz:

```python bio.py --help```

Klasifikáciu je možné spustiť pomocou príkazu:

```python bio.py --mode train test evaluate --save_model ___ --clasified_images_dir data/______.csv --model ___mcs --epochs n --test_images_dir images/___ --train_images_dir images/___ --label_train_file labels/train_labels.csv --label_test_file labels/test_labels.csv```

Kde *--mode* je mód, v ktorom sa má program spustiť. Mód *train* spustí trénovanie modelu, mód *test* spustí testovanie modelu a mód *evaluate* spustí vyhodnotenie modelu. *--save_model* je cesta, kde sa má uložiť model. *--clasified_images_dir* je cesta k súboru, kde sa majú uložiť výsledky klasifikácie jednotivých snímkov. *--model* je model, ktorý sa má použiť. *--epochs* je počet epochov, ktoré sa majú použiť pri trénovaní modelu. *--test_images_dir* je cesta k testovacím snímkam. *--train_images_dir* je cesta k trénovacím snímkam. *--label_train_file* je cesta k súboru s označením trénovacích snímkov. *--label_test_file* je cesta k súboru s označením testovacích snímkov.


## Vyhodnotenie

Trénovanie modelov bolo spustené na počítači s procesorom AMD Ryzen 7 7700x a grafickou kartou NVIDIA GeForce RTX 4070. Trénovanie modelov je ukázané v priečinku *logs/*. V skratke sú výsledky nasledovné:

| Model | Presnosť | Trvanie trénovania | Dataset | Epochy |
| --- | --- | --- | --- | --- |
| ResNet18 | 0.82 | 206 minút | EyePACS | 30 |
| ResNet50 | 0.67 | 270 minút | EyePACS | 30 |
| DenseNet121 | 0.75 | 440 minút | EyePACS | 30 |
| --- | --- | --- | --- | --- |
| ResNet18 | 0.63 | 12 minút | STRaDe | 30 |
| ResNet50 | 0.62 | 17 minút | STRaDe | 30 |
| DenseNet121 | 0.63 | 28 minút | STRaDe | 30 |


## Záver

Tento projekt sa venoval klasifikácii kvality obrazu sietnice prostredníctvom použitia neurónových sietí a pokročilých techník spracovania obrazu. Experimentovali sme s modelmi ResNet18, ResNet50 a DenseNet121 na dvoch rozdielnych datasetoch - EyePACS a STRaDe.

Vyhodnotenie modelov ukázalo, že na datasete EyePACS dosiahli modely pomerne dobré výsledky, s presnosťou pohybujúcou sa od 0.67 do 0.82. Naopak, na datasete STRaDe sme zaznamenali nižšiu presnosť, pričom sa pohybovala od 0.62 do 0.63. Tieto výsledky sa dajú argumentovať nízkym počtom fotiek v datasete STRaDe na trénovanie a testovanie modelov, konkrétne 1188 fotiek na trénovanie a testovanie, pričom dataset bol rozdelený na 730 fotiek na trénovanie a 458 fotiek na testovanie. Naopak, dataset EyePACS obsahoval 12544 fotiek na trénovanie a 16250 fotiek na testovanie.

V budúcnosti je možné vylepšiť výkon modelov napríklad prostredníctvom optimalizácie architektúry sietí, zvýšenia trénovacieho datasetu alebo ďalšieho ladenia parametrov trénovania. Ďalším možným vylepšením je použitie ďalších modelov, napríklad modelov z rodiny EfficientNet a iných. Ďalším možným vylepšením je použitie ďalších techník spracovania obrazu, napríklad použitie techník zlepšenia kvality obrazu, ako je napríklad *Super Resolution*. 



## Použité knižnice

Zoznam použitých knižníc je v súbore *requirements.txt*.