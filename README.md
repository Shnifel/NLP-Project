# Tigrinya News Article Classification

## Setup and installation 

In order to replicate the results obtained, we recommend creating a $\texttt{conda}$ environment as follows

```bash
conda create -n nlp_env python=3.11
conda activate nlp_env
```

and installing all dependencies as:

```bash
conda install pip pytorch torchvision torchaudio -c pytorch
pip install -r requirements.txt
```

To deactivate the environment, run:
```bash
conda deactivate
```

## 2. File structure
```
src                                             
├── models                                      -> Model implementations
|   └── baseline.py                                  Base model implementation
|   └── contrastive_learning.py                      Contrastive learning fine tuning
└── utils                                       -> Data loading and evaluation utilities
|   └── data_utils.py                                Loading and preprocessing datasets for both fine-tunings
|   └── evalute.py                                   Model evaluation metrics
|   └── highlight_text.py                            Attention interpretability
├── train.py                                     -> Train models
└── evaluate.py                                  -> Evaluating models
```

## 3. Replicating results
