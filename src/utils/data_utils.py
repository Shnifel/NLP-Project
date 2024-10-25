import pandas as pd
from datasets import Dataset, load_dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import random

def train_valid_test_split(dataset: Dataset):

    train_test = dataset.train_test_split(0.2, seed=42)
    val_test_dataset = train_test['test']
    val_test = val_test_dataset.train_test_split(0.5, seed=42)
    
    return train_test['train'], val_test['train'], val_test['test']


# Final dataset for binary classification
def preprocess_tir_news():
    dataset = load_dataset("masakhane/masakhanews", "tir")

    def binarize_column(data):
        data['label'] = 1 if data['label'] == 1 else 0 # Entertainment is 1
        return data

    # Apply the binarization using map
    train_data = dataset['train'].map(binarize_column)
    val_data = dataset['validation'].map(binarize_column)
    test_data = dataset['test'].map(binarize_column)

    return train_data, val_data, test_data

def preprocess_amh_news():
    dataset = load_dataset('rasyosef/amharic-news-category-classification')['train'].rename_column('article', 'text')
    return train_valid_test_split(dataset)

    # dataset = load_dataset("masakhane/masakhanews", "amh")

    # remap_dict = {5: 0, 0: 1, 2: 2, 3: 3}

    # # Remap label so one of four labels
    # def remap_category(example):
    #     example['label'] = remap_dict[example['label']]
    #     return example

    # # Apply the remapping to the dataset
    # train_data = dataset['train'].map(remap_category)
    # val_data = dataset['validation'].map(remap_category)
    # test_data = dataset['test'].map(remap_category)

    # return train_data, val_data, test_data


def preprocess_amharic_tigrinya_news():
    
    am_dataset = load_dataset("masakhane/masakhanews", "amh")
    tir_dataset = load_dataset("masakhane/masakhanews", "tir")

    # Combine datasets, assuming they have similar structure
    combined_dataset = {
        'text': am_dataset['train']['text'] + tir_dataset['train']['text'],
        'label': am_dataset['train']['label'] + tir_dataset['train']['label']
    }

    dataset = Dataset.from_dict(combined_dataset)

    # Split into train, validation, and test sets
    train_data, val_data, test_data = train_valid_test_split(dataset)

    return train_data, val_data, test_data

def tokenize_text(dataset, tokenizer, col_name = 'text'):
    
    def tokenize_function(data):
        return tokenizer(data[col_name], padding="max_length", truncation=True)
    dataset = dataset.map(tokenize_function, batched=True)
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label', 'text'])
    return dataset


def preprocess_distil_dataset(fname_eng, fname_tir, fname_labels, student_tokenizer, teacher_tokenizer, label_name = 'Category'):
    with open(fname_eng, 'r', encoding='utf-8') as f:
        english_lines = f.readlines()
    with open(fname_tir, 'r', encoding='utf-8') as f:
        tigrinya_lines = f.readlines()
    
    labels_df = pd.read_csv(fname_labels, sep='\t')
    #label_encoder = LabelEncoder()

    category_mapping = {
        'Business and Economy': 1,
        'Entertainment': 3,
        'Politics': 4,
        'Science and Technology': 0,
        'Sport': 2
    }
    labels_df['category_encoded'] = labels_df[label_name].map(category_mapping)
    #labels_df['label_encoded'] = label_encoder.fit_transform(labels_df[label_name])

    labels_df = labels_df[labels_df['category_encoded'].notna()]
    filtered_indices = labels_df.index
    english_lines = [english_lines[i] for i in filtered_indices]
    tigrinya_lines = [tigrinya_lines[i] for i in filtered_indices]

    
    data = {
        'eng': [line.strip() for line in english_lines],
        'tir': [line.strip() for line in tigrinya_lines],
        'label': labels_df['category_encoded'].tolist()
    }

    dataset = Dataset.from_dict(data)

    def tokenize_function_dual(examples):
        return {
            'tir_input_ids': student_tokenizer(examples['tir'], padding='max_length', truncation=True)['input_ids'],
            'tir_attention_mask': student_tokenizer(examples['tir'], padding='max_length', truncation=True)['attention_mask'],
            'eng_input_ids': teacher_tokenizer(examples['eng'], padding='max_length', truncation=True)['input_ids'],
            'eng_attention_mask': teacher_tokenizer(examples['eng'], padding='max_length', truncation=True)['attention_mask'],
            'label': examples['label']
        }
    
    dataset = dataset.map(tokenize_function_dual, batched=True)
    dataset.set_format(type='torch', columns=['tir_input_ids', 'tir_attention_mask','eng_input_ids', 'eng_attention_mask', 'label'])
    return train_valid_test_split(dataset)

def create_positive_pairs(split_dataset, n_samples=40):
    data = {
        'amh': [],
        'tir': [],
        'tir_positive': [],
        'label': []
    }

    # Group texts by category
    texts_by_category = {}
    for i, category in enumerate(split_dataset['label']):
        if category not in texts_by_category:
            texts_by_category[category] = []
        texts_by_category[category].append({
            'amh': split_dataset['amh'][i].strip(),
            'tir': split_dataset['tir'][i].strip(),
            'index': i
        })
    
    # Create positive pairs
    for category, texts in texts_by_category.items():
        for i, text in enumerate(texts):
            # Find a different text from the same category for the positive pair
            positive_indices = [j for j in range(len(texts)) if j != i]
            if positive_indices:  # If there are other texts in this category
                for _ in range(n_samples):
                    positive_idx = random.choice(positive_indices)
                    positive_text = texts[positive_idx]
                    
                    data['amh'].append(text['amh'])
                    data['tir'].append(text['tir'])
                    data['tir_positive'].append(positive_text['tir'])
                    data['label'].append(category)
    
    return Dataset.from_dict(data)


def preprocess_contrastive_dataset(fname_amh, fname_tir, fname_labels, tokenizer, label_name='Category'):
    with open(fname_amh, 'r', encoding='utf-8') as f:
        amharic_lines = f.readlines()
    with open(fname_tir, 'r', encoding='utf-8') as f:
        tigrinya_lines = f.readlines()
    
    labels_df = pd.read_csv(fname_labels, sep='\t')
    
    category_mapping = {
        'Business and Economy': 1,
        'Entertainment': 3,
        'Politics': 4,
        'Science and Technology': 0,
        'Sport': 2,
        'Health': 5
    }
    labels_df['category_encoded'] = labels_df[label_name].map(category_mapping)

    labels_df = labels_df[labels_df['category_encoded'].notna()]
    filtered_indices = labels_df.index
    amharic_lines = [amharic_lines[i] for i in filtered_indices]
    tigrinya_lines = [tigrinya_lines[i] for i in filtered_indices]

    dataset = Dataset.from_dict({
        'amh': amharic_lines,
        'tir': tigrinya_lines,
        'label': labels_df['category_encoded'].tolist()
    })
    
    train_dataset, val_dataset, test_dataset = train_valid_test_split(dataset)

    train_dataset = create_positive_pairs(train_dataset)
    val_dataset = create_positive_pairs(val_dataset)
    test_dataset = create_positive_pairs(test_dataset)

    
    def tokenize_function_contrastive(examples):
        return {
            'tir_anchor_input_ids': tokenizer(examples['amh'], padding='max_length', truncation=True, return_tensors=None)['input_ids'],
            'tir_anchor_attention_mask': tokenizer(examples['amh'], padding='max_length', truncation=True, return_tensors=None)['attention_mask'],
            'tir_positive_input_ids': tokenizer(examples['tir_positive'], padding='max_length', truncation=True, return_tensors=None)['input_ids'],
            'tir_positive_attention_mask': tokenizer(examples['tir_positive'], padding='max_length', truncation=True, return_tensors=None)['attention_mask'],
            'label': examples['label']
        }
    
    train_dataset = train_dataset.map(tokenize_function_contrastive, batched=True)
    val_dataset = val_dataset.map(tokenize_function_contrastive, batched=True)
    test_dataset = test_dataset.map(tokenize_function_contrastive, batched=True)

    columns=['tir_anchor_input_ids', 'tir_anchor_attention_mask', 'tir_positive_input_ids', 'tir_positive_attention_mask', 'label']
    train_dataset.set_format(type='torch', columns=columns)
    val_dataset.set_format(type='torch', columns=columns)
    test_dataset.set_format(type='torch', columns=columns)

    return train_dataset, val_dataset, test_dataset

def preprocess_eval_dataset(fname_amh, fname_tir, fname_eng, fname_labels, tokenizer, label_name='Category'):
    with open(fname_amh, 'r', encoding='utf-8') as f:
        amharic_lines = f.readlines()
    with open(fname_tir, 'r', encoding='utf-8') as f:
        tigrinya_lines = f.readlines()
    with open(fname_eng, 'r', encoding='utf-8') as f:
        eng_lines = f.readlines()
    
    labels_df = pd.read_csv(fname_labels, sep='\t')
    
    category_mapping = {
        'Entertainment': 1
    }
    labels_df['category_encoded'] = labels_df[label_name].map(category_mapping)

    labels_df = labels_df[labels_df['category_encoded'].notna()]
    filtered_indices = labels_df.index
    amharic_lines = [amharic_lines[i] for i in filtered_indices]
    tigrinya_lines = [tigrinya_lines[i] for i in filtered_indices]
    eng_lines = [eng_lines[i] for i in filtered_indices]


    
    data = {
        'eng': [line.strip() for line in eng_lines],
        'tir': [line.strip() for line in tigrinya_lines],
        'amh': [line.strip() for line in amharic_lines],
        'label': labels_df['category_encoded'].tolist()
    }

    dataset = Dataset.from_dict(data)

    train_dataset, val_dataset, test_dataset = train_valid_test_split(dataset)
    
    def tokenize_function_contrastive(examples):
        return {
            'amh_input_ids': tokenizer(examples['amh'], padding='max_length', truncation=True, return_tensors=None)['input_ids'],
            'amh_attention_mask': tokenizer(examples['amh'], padding='max_length', truncation=True, return_tensors=None)['attention_mask'],
            'tir_input_ids': tokenizer(examples['tir'], padding='max_length', truncation=True, return_tensors=None)['input_ids'],
            'tir_attention_mask': tokenizer(examples['tir'], padding='max_length', truncation=True, return_tensors=None)['attention_mask'],
            'eng': examples['eng'],
            'label': examples['label']
        }
    
    train_dataset = train_dataset.map(tokenize_function_contrastive, batched=True)
    val_dataset = val_dataset.map(tokenize_function_contrastive, batched=True)
    test_dataset = test_dataset.map(tokenize_function_contrastive, batched=True)

    columns = ['amh_input_ids', 'tir_input_ids', 'amh_attention_mask', 'tir_attention_mask', 'eng']
    train_dataset.set_format(type='torch', columns=columns)
    val_dataset.set_format(type='torch', columns=columns)
    test_dataset.set_format(type='torch', columns=columns)

    return train_dataset, val_dataset, test_dataset