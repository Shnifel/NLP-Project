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

def preprocess_tir_news():
    dataset = load_dataset("masakhane/masakhanews", "tir")

    def binarize_column(data):
        data['label'] = 1 if data['label'] == 2 else 0 # Technology is 6
        return data

    # Apply the binarization using map
    train_data = dataset['train'].map(binarize_column)
    val_data = dataset['validation'].map(binarize_column)
    test_data = dataset['test'].map(binarize_column)

    return train_data, val_data, test_data

def preprocess_amharic_tigrinya_news():
    
    am_dataset = load_dataset("masakhane/masakhanews", "amh")
    tir_dataset = load_dataset("masakhane/masakhanews", "tir")
    
    # Convert to pandas DataFrames for easier inspection
    # am_df = pd.DataFrame(am_dataset['train'])
    # tir_df = pd.DataFrame(tir_dataset['train'])

    # # Print out the heads of the datasets
    # print("Amharic Dataset Head:")
    # print(am_df.head())
    # print("\nTigrinya Dataset Head:")
    # print(tir_df.head())

    # Combine datasets, assuming they have similar structure
    combined_dataset = {
        'text': am_dataset['train']['text'] + tir_dataset['train']['text'],
        'label': am_dataset['train']['label'] + tir_dataset['train']['label']
    }

    dataset = Dataset.from_dict(combined_dataset)

    # Split into train, validation, and test sets
    train_data, val_data, test_data = train_valid_test_split(dataset)
    
    # print("\nTrain Data Size:", len(train_data))
    # print("Validation Data Size:", len(val_data))
    # print("Test Data Size:", len(test_data))

    return train_data, val_data, test_data

def tokenize_text(dataset, tokenizer, col_name = 'text'):
    
    def tokenize_function(data):
        return tokenizer(data[col_name], padding="max_length", truncation=True)
    dataset = dataset.map(tokenize_function, batched=True)
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label', 'text'])
    return dataset

def tokenize_text_for_contrastive(dataset, tokenizer, text_col):
    def tokenize_function(data):
        return tokenizer(data[text_col], padding="max_length", truncation=True)
    
    dataset = dataset.map(tokenize_function, batched=True)
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label', text_col])
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
        'Sport': 2
    }
    labels_df['category_encoded'] = labels_df[label_name].map(category_mapping)

    labels_df = labels_df[labels_df['category_encoded'].notna()]
    filtered_indices = labels_df.index
    amharic_lines = [amharic_lines[i] for i in filtered_indices]
    tigrinya_lines = [tigrinya_lines[i] for i in filtered_indices]

    # Create a dictionary to store texts by category
    texts_by_category = {}
    for i, category in enumerate(labels_df['category_encoded'].tolist()):
        if category not in texts_by_category:
            texts_by_category[category] = []
        texts_by_category[category].append({
            'amh': amharic_lines[i].strip(),
            'tir': tigrinya_lines[i].strip(),
            'index': i
        })

    # Create positive pairs within the same category
    data = {
        'amh': [],
        'tir': [],
        'tir_positive': [],  # This will store the positive pair for Tigrinya
        'label': []
    }

    for category, texts in texts_by_category.items():
        for i, text in enumerate(texts):
            # Find a different text from the same category for the positive pair
            positive_indices = [j for j in range(len(texts)) if j != i]
            if positive_indices:  # If there are other texts in this category
                positive_idx = random.choice(positive_indices)
                positive_text = texts[positive_idx]
                
                data['amh'].append(text['amh'])
                data['tir'].append(text['tir'])
                data['tir_positive'].append(positive_text['tir'])
                data['label'].append(category)

    contrastive_dataset = Dataset.from_dict(data)

    # Split the dataset into train and temp (validation + test) sets
    train_size = 0.8
    train_dataset = contrastive_dataset.train_test_split(test_size=1 - train_size)['train']
    temp_dataset = contrastive_dataset.train_test_split(test_size=1 - train_size)['test']

    # Further split temp_dataset into validation and test sets
    val_size = 0.5
    val_dataset = temp_dataset.train_test_split(test_size=1 - val_size)['train']
    test_dataset = temp_dataset.train_test_split(test_size=1 - val_size)['test']
    
    def tokenize_function_contrastive(examples):
        return {
            'tir_anchor_input_ids': tokenizer(examples['tir'], padding='max_length', truncation=True, return_tensors=None)['input_ids'],
            'tir_anchor_attention_mask': tokenizer(examples['tir'], padding='max_length', truncation=True, return_tensors=None)['attention_mask'],
            'tir_positive_input_ids': tokenizer(examples['tir_positive'], padding='max_length', truncation=True, return_tensors=None)['input_ids'],
            'tir_positive_attention_mask': tokenizer(examples['tir_positive'], padding='max_length', truncation=True, return_tensors=None)['attention_mask'],
            'label': examples['label']
        }
    
    train_dataset = train_dataset.map(tokenize_function_contrastive, batched=True)
    val_dataset = val_dataset.map(tokenize_function_contrastive, batched=True)
    test_dataset = test_dataset.map(tokenize_function_contrastive, batched=True)

    train_dataset.set_format(type='torch', columns=['tir_anchor_input_ids', 'tir_anchor_attention_mask', 'tir_positive_input_ids', 'tir_positive_attention_mask', 'label'])
    val_dataset.set_format(type='torch', columns=['tir_anchor_input_ids', 'tir_anchor_attention_mask', 'tir_positive_input_ids', 'tir_positive_attention_mask', 'label'])
    test_dataset.set_format(type='torch', columns=['tir_anchor_input_ids', 'tir_anchor_attention_mask', 'tir_positive_input_ids', 'tir_positive_attention_mask', 'label'])

    return train_dataset, val_dataset, test_dataset