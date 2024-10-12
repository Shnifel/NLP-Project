import pandas as pd
from datasets import Dataset, load_dataset

def train_valid_test_split(dataset: Dataset):

    train_test = dataset.train_test_split(0.2, seed=42)
    val_test_dataset = train_test['test']
    val_test = val_test_dataset.train_test_split(0.5, seed=42)
    
    return train_test['train'], val_test['train'], val_test['test']

def preprocess_amharic_news():
    dataset = load_dataset("masakhane/masakhanews", "tir")

    def binarize_column(data):
        data['label'] = 1 if data['label'] == 2 else 0 # Sports is 5
        return data

    # Apply the binarization using map
    train_data = dataset['train'].map(binarize_column)
    val_data = dataset['validation'].map(binarize_column)
    test_data = dataset['test'].map(binarize_column)

    return train_data, val_data, test_data

def tokenize_text(dataset, tokenizer):
    
    def tokenize_function(data):
        return tokenizer(data["text"], padding="max_length", truncation=True)
    dataset = dataset.map(tokenize_function, batched=True)
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label', 'text'])
    return dataset
