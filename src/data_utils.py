import pandas as pd
from datasets import Dataset, load_dataset


def construct_data_loaders(data_path):
     # Load the dataset
    dataset = load_dataset("rasyosef/amharic-news-category-classification")

    # Access the 'train' split (or any other split as per your requirement)
    train_split = dataset['train']
    
    # Convert the 'train' split to a pandas DataFrame
    df = train_split.to_pandas()
    
    # Display the first few rows to verify
    print("Initial DataFrame Head:")
    print(df.head())

    # Strip leading and trailing whitespace from 'article' and 'category' columns
    df['article'] = df['article'].astype(str).str.strip()
    df['label'] = df['label'].astype(str).str.strip()

    # Encode categories
    # Assuming 'category' has integer labels and you're converting it to binary
    # Here, replace `2` with the appropriate category you want to encode as 1
    df['category_encoded'] = df['label'].apply(lambda x: 1 if x == '2' else 0)


    sampled_dataset = Dataset.from_pandas(df[['article', 'category_encoded']])

    # Split the dataset into train and test sets (90% train, 10% test)
    train_test = sampled_dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = train_test['train']
    test_dataset = train_test['test']

    # Rename columns to match expected format for most NLP models
    train_dataset = train_dataset.rename_column("article", "text")
    train_dataset = train_dataset.rename_column("category_encoded", "label")
    
    test_dataset = test_dataset.rename_column("article", "text")
    test_dataset = test_dataset.rename_column("category_encoded", "label")

    # Set the format of the datasets to PyTorch tensors (optional, based on your framework)
    # train_dataset.set_format(type='torch', columns=['text', 'label'])
    # test_dataset.set_format(type='torch', columns=['text', 'label'])

    return train_dataset, test_dataset

def tokenize_text(dataset, tokenizer):
    
    def tokenize_function(data):
        return tokenizer(data["text"], padding="max_length", truncation=True)
    dataset = dataset.map(tokenize_function, batched=True)
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    return dataset
