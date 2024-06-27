# Project-Representation-Learning

Whole project is in Polish language.

# Projekt Uczenie Reprezentacji - Uczenie Reprezentacji na danych medycznych

Wiktor Sadowy | Tomasz Hałas

## Linki
[Dataset >>](https://archive.ics.uci.edu/dataset/544/estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition)  
[Google Drive (dane do odpalenia dvc repro)](https://drive.google.com/drive/folders/1kgBpQnbrhJJSH-sBiTvAxMxAiX3oS0E3?usp=sharing)  
[Opis danych >>](https://luna16.grand-challenge.org/)  

## Uruchomienie
```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
dvc init
dvc repro
```

Notebooki zostaną wykonane i zapisane w `results/` jako pliki HTML. Może to chwilę potrwać.

Uwaga: przed uruchomieniem `dvc repro` należy pobrać cały dataset z linku wyżej i umieścić go w folderze projektu

## Porównanie wyników

| Model              | OPIS                                            | Accuracy | Precision | Recall  | F1 Score |
|--------------------|-------------------------------------------------|----------|-----------|---------|----------|
| StandardClassifier | Klasyfikator bez użycia self-supervised         | 0.86794  | 0.62549   | 0.51057 | 0.56222  |
| BYOL               | Użycie projection w online i target enkoderze   | 0.83581  | 0.53608   | 0.08455 | 0.14607  |
| DINO               | Architektura student-nauczyciel                 | 0.83391  | 0         | 0       | 0        |
| MoCo               | Użycie kolejki do przechowywania użytych batchy | 0.88740  | 0.68172   | 0.49106 | 0.57088  |
