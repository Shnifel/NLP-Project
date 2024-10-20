#!/bin/bash

## Baseline
python src/train.py

## First finetuning
python src/train.py --model_run first-finetuning --save_path ./checkpoints/finetuned/ --run_name first-finetuning                                                                                                                

## Contrastive learning
python src/train.py --model_run contrastive --save_path ./checkpoints/contrastive/ --run_name contrastive --checkpoint-path ./checkpoints/finetuned/checkpoint-14994

## Second finetuning
python src/train.py --model_run baseline --save_path ./checkpoints/final/ --run_name final --model_name ./checkpoints/finetuned/checkpoint-14994 --epochs 15 --checkpoint-path ./checkpoints/contrastive --learning-rate 5e-3