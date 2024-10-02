from IPython.display import clear_output
import csv
import pandas as pd
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('SwahiliNewsClassificationDataset.csv')

df['content'] = df['content'].str.strip()
df['category'] = df['category'].str.strip()

# encodes categories
label_encoder = LabelEncoder()
df['category_encoded'] = label_encoder.fit_transform(df['category'])

# test on a small subset of data first ti test the pipeline
sampled_df = df.sample(n=100, random_state=42) 

sampled_dataset = Dataset.from_pandas(sampled_df[['content', 'category_encoded']])
# dataset = Dataset.from_pandas(df[['content', 'category_encoded']])

train_test = sampled_dataset.train_test_split(test_size=0.1)
train_dataset = train_test['train']
test_dataset = train_test['test']

train_dataset = train_dataset.rename_column("content", "text")
train_dataset = train_dataset.rename_column("category_encoded", "label")

test_dataset = test_dataset.rename_column("content", "text")
test_dataset = test_dataset.rename_column("category_encoded", "label")

batch_size = 8

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

#Loading BERT for now as test, need to decide on a pretrained african language mdoel
model = AutoModelForSequenceClassification.from_pretrained("bert-base-multilingual-cased", num_labels=len(label_encoder.classes_))

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
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
    eval_dataset=test_dataset,
)

trainer.train()