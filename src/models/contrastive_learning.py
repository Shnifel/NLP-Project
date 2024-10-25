from transformers import AutoTokenizer, AutoModel
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_scheduler
from tqdm.auto import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
from models.baseline import NewsClassificationModel
from tqdm import tqdm
import os
from copy import deepcopy
import wandb

class ContrastiveNet:
    def __init__(self, model_name, train_dataset, val_dataset, test_dataset, batch_size=32, checkpoint_path = None):
        # Initialize the base model without classification head
        self.base_model = NewsClassificationModel(
            model_name=model_name,
            tokenizer_name=model_name if checkpoint_path is None else checkpoint_path,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            checkpoint_path=checkpoint_path,
            tokenize=False,
            use_lora= True,
            is_contrastive=True,
            lora_checkpoint=True
        )

        # Custom collate function for batching
        # basically does padding (which might not be good) because of different sized tensors..
        def collate_fn(batch):
            # Find max length in the batch for anchor and positive
            max_len_anchor = max(len(item['tir_anchor_input_ids']) for item in batch)
            max_len_positive = max(len(item['tir_positive_input_ids']) for item in batch)
            
            # Initialize tensors
            batch_size = len(batch)
            anchor_input_ids = torch.zeros((batch_size, max_len_anchor), dtype=torch.long)
            anchor_attention_mask = torch.zeros((batch_size, max_len_anchor), dtype=torch.long)
            positive_input_ids = torch.zeros((batch_size, max_len_positive), dtype=torch.long)
            positive_attention_mask = torch.zeros((batch_size, max_len_positive), dtype=torch.long)
            labels = torch.zeros(batch_size, dtype=torch.long)
            
            # Fill tensors
            for i, item in enumerate(batch):
                # Anchor
                anc_len = len(item['tir_anchor_input_ids'])
                anchor_input_ids[i, :anc_len] = torch.tensor(item['tir_anchor_input_ids'])
                anchor_attention_mask[i, :anc_len] = torch.tensor(item['tir_anchor_attention_mask'])
                
                # Positive
                pos_len = len(item['tir_positive_input_ids'])
                positive_input_ids[i, :pos_len] = torch.tensor(item['tir_positive_input_ids'])
                positive_attention_mask[i, :pos_len] = torch.tensor(item['tir_positive_attention_mask'])
                
                # Label
                labels[i] = item['label']
            
            return {
                'tir_anchor_input_ids': anchor_input_ids,
                'tir_anchor_attention_mask': anchor_attention_mask,
                'tir_positive_input_ids': positive_input_ids,
                'tir_positive_attention_mask': positive_attention_mask,
                'label': labels
            }

        # Create dataloaders with custom collate function
        self.train_dataloader = DataLoader(
            train_dataset, 
            shuffle=True, 
            batch_size=batch_size, 
            # collate_fn=collate_fn
        )
        self.val_dataloader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            # collate_fn=collate_fn
        )
        self.test_dataloader = DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            # collate_fn=collate_fn
        )
        
        # Add temperature parameter for InfoNCE loss
        self.temperature = 0.07

    # infoNCE loss (might need to change this because its used in self-superised learning)
    def info_nce_loss(self, anchor_features, positive_features):
        # Normalize the features
        anchor_features = F.normalize(anchor_features, dim=1)
        positive_features = F.normalize(positive_features, dim=1)
        
        # Compute similarity matrix
        logits = torch.matmul(anchor_features, positive_features.T) / self.temperature
        
        # Labels are all diagonal elements (positives)
        batch_size = anchor_features.shape[0]
        labels = torch.arange(batch_size, device=anchor_features.device)
        
        # Compute cross entropy loss
        loss = F.cross_entropy(logits, labels)
        
        return loss

    def train_contrastive(self, lr, n_epochs, save_path, run_name):
        prev_val_loss = np.inf

        wandb.init(project="huggingface")
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.base_model.model.to(device)
        
        # Initialize optimizer and scheduler
        optimizer = AdamW(self.base_model.model.parameters(), lr=lr)
        num_training_steps = n_epochs * len(self.train_dataloader)
        lr_scheduler = get_scheduler(
            "linear",
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=num_training_steps
        )
        
        # Training loop
        progress_bar = tqdm(range(num_training_steps))
        self.base_model.model.train()
        
        train_losses = []

        for epoch in range(n_epochs):
            epoch_loss = 0
            num_batches = 0
            self.base_model.model.train()
            print("HERE")
            
            for batch in self.train_dataloader:
                # Move batch to device
                batch = {k: v.to(device) for k, v in batch.items()}

                # Get embeddings for anchor and positive samples
                anchor_outputs = self.base_model.model(
                    input_ids=batch['tir_anchor_input_ids'],
                    attention_mask=batch['tir_anchor_attention_mask'],
                    output_hidden_states=True
                )
                
                positive_outputs = self.base_model.model(
                    input_ids=batch['tir_positive_input_ids'],
                    attention_mask=batch['tir_positive_attention_mask'],
                    output_hidden_states=True
                )

                # Use [CLS] token embedding (last hidden state)
                anchor_embeddings = anchor_outputs.hidden_states[-1][:, 0]
                positive_embeddings = positive_outputs.hidden_states[-1][:, 0]

                # Calculate InfoNCE loss
                loss = self.info_nce_loss(anchor_embeddings, positive_embeddings)
                
                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                
                epoch_loss += loss.item()
                num_batches += 1
                progress_bar.update(1)
            
            avg_epoch_loss = epoch_loss / num_batches
            train_losses.append(avg_epoch_loss)
            print(f"Epoch {epoch + 1}/{n_epochs}, Average Loss: {avg_epoch_loss:.4f}")

            val_loss = self.evaluate()['validation_loss']

            if val_loss < prev_val_loss:
                prev_val_loss = val_loss
                save_model = deepcopy(self.base_model.model)
                save_model.merge_and_unload().save_pretrained(save_path)
            
            
            wandb.log({
                "InfoNCE Loss": avg_epoch_loss,
                "InfoNCE Validation": val_loss,
                "Epoch": epoch
            })

        # Save the trained model
        return train_losses

    def evaluate(self):
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.base_model.model.eval()
        self.base_model.model.to(device)
        
        val_losses = []
        
        with torch.no_grad():
            for batch in self.val_dataloader:
                batch = {k: v.to(device) for k, v in batch.items()}
                
                # Get embeddings
                anchor_outputs = self.base_model.model(
                    input_ids=batch['tir_anchor_input_ids'],
                    attention_mask=batch['tir_anchor_attention_mask'],
                    output_hidden_states=True
                )
                
                positive_outputs = self.base_model.model(
                    input_ids=batch['tir_positive_input_ids'],
                    attention_mask=batch['tir_positive_attention_mask'],
                    output_hidden_states=True
                )

                # Use [CLS] token embedding
                anchor_embeddings = anchor_outputs.hidden_states[-1][:, 0]
                positive_embeddings = positive_outputs.hidden_states[-1][:, 0]

                # Calculate validation loss
                loss = self.info_nce_loss(anchor_embeddings, positive_embeddings)
                val_losses.append(loss.item())

        avg_val_loss = sum(val_losses) / len(val_losses)
        return {
            'validation_loss': avg_val_loss
        }
    
    def plot_tsne(self, stage = "before"):

        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.base_model.model.eval()
        self.base_model.model.to(device)
        
        anchors = []
        positives = []
        labels = []
        
        with torch.no_grad():
            for batch in tqdm(self.test_dataloader):
                batch = {k: v.to(device) for k, v in batch.items()}
                
                # Get embeddings
                anchor_outputs = self.base_model.model(
                    input_ids=batch['tir_anchor_input_ids'],
                    attention_mask=batch['tir_anchor_attention_mask'],
                    output_hidden_states=True
                )
                
                positive_outputs = self.base_model.model(
                    input_ids=batch['tir_positive_input_ids'],
                    attention_mask=batch['tir_positive_attention_mask'],
                    output_hidden_states=True
                )

                # Use [CLS] token embedding
                anchor_embeddings = anchor_outputs.hidden_states[-1][:, 0]
                positive_embeddings = positive_outputs.hidden_states[-1][:, 0]


                anchors.append(anchor_embeddings.cpu().numpy())
                
                positives.append(positive_embeddings.cpu().numpy())
                labels.append(batch['label'].clone().cpu().numpy())


        anchors = np.concatenate(anchors, axis=0)
        positives = np.concatenate(positives, axis=0)
        labels = np.concatenate(labels, axis=0)

        print(anchors.shape)
        print(labels.shape)
        # Combine anchors and positives for t-SNE
        all_embeddings = np.vstack([anchors, positives])

        print(all_embeddings.shape)

        # Apply t-SNE
        tsne = TSNE(n_components=2, random_state=42)
        reduced_embeddings = tsne.fit_transform(all_embeddings)

        classes = ['Science and Technology', 'Business and Economy','Sport', 'Entertainment', 'Politics', 'Health' ]
        
        
        # Create a scatter plot with different shapes
        fig, ax = plt.subplots()

        # Plot anchors
        ax.scatter(reduced_embeddings[:len(anchors), 0], reduced_embeddings[:len(anchors), 1], 
                c=labels, cmap = 'viridis', marker='o', label='Anchors')

        # Plot positives
        scatter = ax.scatter(reduced_embeddings[len(anchors):, 0], reduced_embeddings[len(anchors):, 1], 
                c=labels, cmap = 'viridis', marker='x', label='Positives')

        # Add labels, legend, and title
        ax.set_xlabel("t-SNE Dim 1")
        ax.set_ylabel("t-SNE Dim 2")
        ax.legend(handles=scatter.legend_elements()[0], labels=classes)

        save_path = f"./results/contrastive/"
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(f"{save_path}tsne_{stage}.pdf")