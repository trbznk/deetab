# deetab

Looking for good solutions to apply deep learning on tabular data.

Current Datasets:

[ ] [Kaggle Tabular Playground Series May 2022](https://www.kaggle.com/competitions/tabular-playground-series-may-2022)

## Quick Start

```console
python deetab.py
INFO: read data from cache
(1) TRAIN: 0.30 DEV: 0.17
(2) TRAIN: 0.14 DEV: 0.13
(3) TRAIN: 0.12 DEV: 0.12
(4) TRAIN: 0.11 DEV: 0.12
(5) TRAIN: 0.10 DEV: 0.12
(6) TRAIN: 0.10 DEV: 0.11
(7) TRAIN: 0.09 DEV: 0.11
...
```

## Network Architecture

```console
DeeTab(
  (m): Sequential(
    (0): Linear(in_features=41, out_features=512, bias=True)
    (1): SiLU()
    (2): Linear(in_features=512, out_features=384, bias=True)
    (3): SiLU()
    (4): Linear(in_features=384, out_features=256, bias=True)
    (5): SiLU()
    (6): Linear(in_features=256, out_features=128, bias=True)
    (7): SiLU()
    (8): Linear(in_features=128, out_features=64, bias=True)
    (9): SiLU()
    (10): Linear(in_features=64, out_features=1, bias=True)
    (11): Sigmoid()
  )
)
```
