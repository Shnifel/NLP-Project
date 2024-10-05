import pandas as pd
from datasets import Dataset


def construct_data_loaders(data_path):
    df = pd.read_csv(data_path)

    df['content'] = df['content'].str.strip()
    df['category'] = df['category'].str.strip()

    # encodes categories
    df['category_encoded'] = df['category'].apply(lambda x: 1 if x == "uchumi" else 0)

    # test on a small subset of data first ti test the pipeline
    sampled_df = df.sample(n=10000, random_state=42) 
    sampled_dataset = Dataset.from_pandas(sampled_df[['content', 'category_encoded']])
    # dataset = Dataset.from_pandas(df[['content', 'category_encoded']])

    train_test = sampled_dataset.train_test_split(test_size=0.1)
    train_dataset = train_test['train']
    test_dataset = train_test['test']

    train_dataset = train_dataset.rename_column("content", "text")
    train_dataset = train_dataset.rename_column("category_encoded", "label")

    test_dataset = test_dataset.rename_column("content", "text")
    test_dataset = test_dataset.rename_column("category_encoded", "label")

    return train_dataset, test_dataset


def tokenize_text(dataset, tokenizer):
    
    def tokenize_function(data):
        return tokenizer(data["text"], padding="max_length", truncation=True)
    dataset = dataset.map(tokenize_function, batched=True)
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    return dataset
