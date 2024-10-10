import pandas as pd
from datasets import Dataset, load_dataset

def train_valid_test_split(dataset: Dataset):

    train_test = dataset.train_test_split(0.2, seed=42)
    val_test_dataset = train_test['test']
    val_test = val_test_dataset.train_test_split(0.5, seed=42)
    
    return train_test['train'], val_test['train'], val_test['test']

def preprocess_amharic_news():
    dataset = load_dataset("rasyosef/amharic-news-category-classification")
    train_split = dataset['train']
    df = train_split.to_pandas()
    
    df['article'] = df['article'].astype(str).str.strip()
    df['label'] = df['label'].astype(str).str.strip()
    df['category_encoded'] = df['label'].apply(lambda x: 1 if x == '2' else 0)

    dataset = Dataset.from_pandas(df[['article', 'category_encoded']])
    dataset = dataset.rename_column('article', 'text')
    dataset = dataset.rename_column('category_encoded', 'label')

    return train_valid_test_split(dataset)

def tokenize_text(dataset, tokenizer):
    
    def tokenize_function(data):
        return tokenizer(data["text"], padding="max_length", truncation=True)
    dataset = dataset.map(tokenize_function, batched=True)
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label', 'text'])
    return dataset
