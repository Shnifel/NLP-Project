from transformers import AutoTokenizer, AutoModel
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_scheduler
from tqdm.auto import tqdm
import torch.nn.functional as F


class ContrastiveLearningModel:

    def __init__(self, model_name=None, train_dataset=None, batch_size=32):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)

    def contrastive_loss(self, anchor, positive, negative, temperature=0.1):
        # Normalize embeddings
        anchor = F.normalize(anchor, dim=-1)
        positive = F.normalize(positive, dim=-1)
        negative = F.normalize(negative, dim=-1)
        
        # Compute cosine similarity
        pos_sim = torch.matmul(anchor, positive.T) / temperature
        neg_sim = torch.matmul(anchor, negative.T) / temperature

        # Labels for contrastive loss: 1 for positive, 0 for negative
        labels = torch.arange(anchor.size(0)).long().to(anchor.device)
        
        # Concatenate positive and negative similarities and calculate loss
        logits = torch.cat([pos_sim, neg_sim], dim=1)
        return F.cross_entropy(logits, labels)

    def train(self, lr, n_epochs, temperature=0.1):
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model.to(device)
        optimizer = AdamW(self.model.parameters(), lr=lr)

        num_training_steps = n_epochs * len(self.train_dataloader)
        lr_scheduler = get_scheduler(
            "linear",
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=num_training_steps
        )

        progress_bar = tqdm(range(num_training_steps))

        self.model.train()
        for epoch in range(n_epochs):
            for batch in self.train_dataloader:
                # Tokenize text pairs (positive and negative examples)
                anchor_input = self.tokenizer(batch['anchor_text'], return_tensors='pt', padding=True, truncation=True).to(device)
                positive_input = self.tokenizer(batch['positive_text'], return_tensors='pt', padding=True, truncation=True).to(device)
                negative_input = self.tokenizer(batch['negative_text'], return_tensors='pt', padding=True, truncation=True).to(device)
                
                # Get model embeddings
                anchor_emb = self.model(**anchor_input).last_hidden_state[:, 0, :]  # CLS token
                positive_emb = self.model(**positive_input).last_hidden_state[:, 0, :]
                negative_emb = self.model(**negative_input).last_hidden_state[:, 0, :]
                
                # Compute contrastive loss
                loss = self.contrastive_loss(anchor_emb, positive_emb, negative_emb, temperature)

                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                progress_bar.update(1)

            print(f"Epoch {epoch}, Loss: {loss.item()}")

    def evaluate(self):
        
        pass
