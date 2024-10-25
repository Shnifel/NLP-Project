# Tigrinya News Article Classification

## 1. Setup and installation 

In order to replicate the results obtained, we recommend creating a $\texttt{conda}$ environment as follows

```bash
conda create -n nlp_env
conda activate nlp_env
```

and installing all dependencies as:

```bash
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

All models are trained from ```train.py```, and has the following arguments:

- `--model_run` (default: `'baseline'`): Specifies the model run type. One of `['baseline', 'first-finetuning', 'contrastive']`
- `--epochs` (default: `15`): Number of epochs to train for
- `--batch_size` (default: `16`): Batch size for training
- `--save_path` (default: `'./checkpoints/baseline/'`): Directory to save checkpoints
- `--learning-rate` (default: `0.0005`): Learning rate for training
- `--checkpoint-path` (default: `None`): Path to a checkpoint to load the model from
- `--model_name` (default: `"fgaim/tielectra-small"`): Name of the pre-trained model to load
- `--run_name` (default: `'baseline'`): Name for the current run (used for logging and checkpoint naming)

### Baseline

```bash
python src/train.py
```
This will finetune the TiElectra Model on the Masakhane News dataset directly

### Model finetunings

To run the masked language modelling
```bash
python src/train.py --model_run first-finetuning --save_path ./amh_finetuned/ --run_name masked_lm
```

To run the contrastive learning
```bash
python src/train.py --model_run contrastive --save_path ./checkpoints/contrastive_masked/ --run_name contrastive --checkpoint-path .\checkpoints\amh_finetuned\<best-checkpoint-saved>\ --epochs 10
```

To then run the second and final finetuning on the downstream task:
```bash
python src/train.py --model_run baseline --save_path ./checkpoints/final/ --run_name final --model_name ./checkpoints/contrastive_masked/ --epochs 15 --checkpoint-path ./checkpoints/contrastive_masked --learning-rate 5e-2
```

### Evalution
```bash
python src/eval.py --checkpoint-path <checkpoint-path> --save_path ./results/baseline --lora_checkpoint True   
```




