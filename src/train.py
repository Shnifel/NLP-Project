from IPython.display import clear_output
import csv
import pandas as pd
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.preprocessing import LabelEncoder
from src.data_utils import construct_data_loaders
from src.evaluate_utils import compute_metrics


def train_model(model, train_dataset, eval_dataset, batch_size, num_epochs = 3):
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs= num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir='./logs',
        evaluation_strategy="epoch",
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics
    )

    trainer.train()