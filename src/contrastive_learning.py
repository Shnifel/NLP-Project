from transformers import AutoTokenizer, AutoModel
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_scheduler
from tqdm.auto import tqdm
import torch.nn.functional as F
from src.data_utils import tokenize_text
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments, EvalPrediction
from peft import LoraConfig, TaskType, get_peft_model
from peft.peft_model import PeftModel
from src.evaluate_utils import compute_metrics
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, cohen_kappa_score
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import numpy as np
from src.baseline import NewsClassificationModel
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import get_scheduler
from tqdm.auto import tqdm

class ContrastiveNet:

    def __init__(self, model_name, train_dataset, val_dataset, test_dataset, batch_size=32):
        # Define a single model for both Amharic and Tigrinya text
        self.model = NewsClassificationModel(model_name, model_name, train_dataset, val_dataset, test_dataset, 
                                             tokenize=False, n_labels=None)

        self.train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
        self.val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
        self.test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    def train_contrastive(self, lr, n_epochs, margin=0.5):
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model.model.to(device)
        
        # Optimizer and scheduler
        optimizer = torch.optim.AdamW(self.model.model.parameters(), lr=lr)
        num_training_steps = n_epochs * len(self.train_dataloader)
        lr_scheduler = get_scheduler(
            "linear",
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=num_training_steps
        )
        
        # Contrastive Loss (can use other loss funcs)
        contrastive_loss = nn.CosineEmbeddingLoss(margin=margin)

        # Training loop
        progress_bar = tqdm(range(num_training_steps))
        self.model.model.train()

        for epoch in range(n_epochs):
            for batch in self.train_dataloader:
                # Move batch to device
                batch = {k: v.to(device) for k, v in batch.items()}

                # Forward pass: Amharic (anchor) and Tigrinya (positive)
                amharic_outputs = self.model.model(
                    input_ids=batch['anchor_text'],
                    attention_mask=batch['anchor_attention_mask']
                ).last_hidden_state.mean(dim=1)  # Take the mean of the hidden states as embeddings

                tigrinya_outputs = self.model.model(
                    input_ids=batch['positive_text'],
                    attention_mask=batch['positive_attention_mask']
                ).last_hidden_state.mean(dim=1)

                # Targets for contrastive loss (1 for similar pairs)
                targets = torch.ones(amharic_outputs.size(0)).to(device)

                # Compute contrastive loss 
                loss = contrastive_loss(amharic_outputs, tigrinya_outputs, targets)

                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                lr_scheduler.step()

                progress_bar.update(1)

            print(f"Epoch {epoch}, Loss: {loss.item()}")

        # Save the trained model
        self.model.model.save_pretrained('./contrastive_model')

    def evaluate(self):
        # TODO: evaluation embeddings
        pass