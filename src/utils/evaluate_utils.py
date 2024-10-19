from IPython.display import clear_output
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import Trainer
from sklearn.metrics import confusion_matrix
import matplotlib.font_manager as font_manager
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments, EvalPrediction
import matplotlib
import evaluate
import numpy as np

# Load metrics
accuracy_metric = evaluate.load("accuracy")
precision_metric = evaluate.load("precision")
recall_metric = evaluate.load("recall")
f1_metric = evaluate.load("f1")
cohens_kappa_metric =  evaluate.load("Cohen's Kappa")


def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    labels = p.label_ids
    
    # Compute accuracy, precision, recall, and F1
    accuracy = accuracy_metric.compute(predictions=preds, references=labels)
    precision = precision_metric.compute(predictions=preds, references=labels, average="binary")
    recall = recall_metric.compute(predictions=preds, references=labels, average="binary")
    f1 = f1_metric.compute(predictions=preds, references=labels, average="binary")
    cohens_kappa = cohens_kappa_metric.compute(predictions=preds, references=labels, average="binary")
    
    return {
        'accuracy': accuracy['accuracy'],
        'precision': precision['precision'],
        'recall': recall['recall'],
        'f1': f1['f1'],
        'cohens_kappa': cohens_kappa["Cohen's Kappa"]
    }


class ModelEvaluator:

    def __init__(self, model, save_path):

        self.model = model # Model to evaluate
        self.save_path = save_path # Path to save results to
        os.makedirs(self.save_path, exist_ok=True)

        self.trainer = Trainer(
            model=self.model,
            compute_metrics=compute_metrics
        )

        # Configure matplotlib to use a font that supports Amharic characters
        font_path = "C:/Users/Muhammad Sahal/Downloads/Noto_Sans_Ethiopic/static/NotoSansEthiopic-Regular.ttf"  # Update this path
        # font_path = "C:/Users/chris/Downloads/Noto_Sans_Ethiopic/static/NotoSansEthiopic-Regular.ttf"  # Update this path
        self.font_prop = font_manager.FontProperties(fname=font_path)
        matplotlib.rcParams['font.family'] = self.font_prop.get_name()

    def evaluate_classification(self, test_dataset):

        # Get predictions
        preds = self.trainer.predict(test_dataset)
        metrics = preds.metrics

        with open(os.path.join(self.save_path, "evaluation_metrics.txt"), "w", encoding='utf-8') as f:
            f.write(metrics)

        # Compute confusion matrix
        cm = confusion_matrix(preds.label_ids, preds.predictions)
        
        # Save confusion matrix
        plt.figure(figsize=(8,6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title('Confusion Matrix')
        plt.savefig(os.path.join(self.save_path, "confusion_matrix.pdf"))
        plt.close()

    def visualize_attention(self, test_dataset, selected_indices):
         # For selected indices, visualize attention and highlight relevant text
        for idx in selected_indices:
            # Get input features
            sample = test_dataset[idx]

            # Convert to tensors and add batch dimension
            input_ids = torch.tensor(sample['input_ids'], dtype=torch.long).unsqueeze(0).to(self.model.device)
            attention_mask = torch.tensor(sample['attention_mask'], dtype=torch.long).unsqueeze(0).to(self.model.device)

            inputs = {'input_ids': input_ids, 'attention_mask': attention_mask}

            with torch.no_grad():
                outputs = self.model(**inputs, output_attentions=True)

            attentions = outputs.attentions  # Tuple of (num_layers, batch_size, num_heads, seq_len, seq_len)

            # Get tokens
            tokens = self.tokenizer.convert_ids_to_tokens(sample['input_ids'])

            # Clean up tokens (exclude CLS token)
            tokens_cleaned = []
            for token in tokens[1:]:
                if token in ['[SEP]', '[PAD]']:
                    tokens_cleaned.append(token)
                elif token.startswith('##') or token.startswith('Ġ'):
                    tokens_cleaned.append(token.replace('##', '').replace('Ġ', ''))
                else:
                    tokens_cleaned.append(token)

            # Limit sequence length
            max_seq_len = 30
            seq_len = min(len(tokens_cleaned), max_seq_len)
            tokens_cleaned = tokens_cleaned[:seq_len]

            # Extract attention from last layer
            last_layer_attention = attentions[-1][0]  # Shape: (num_heads, seq_len_full, seq_len_full)

            # 1. Plot CLS Token Attention to Other Tokens
            # CLS token attention to all tokens (including CLS token)
            cls_attention_heads = last_layer_attention[:, 0, :]  # Shape: (num_heads, seq_len_full)

            # Exclude CLS token from attention and tokens
            cls_attention_heads = cls_attention_heads[:, 1:]  # Shape: (num_heads, seq_len_full -1)
            cls_attention_heads = cls_attention_heads[:, :seq_len]  # Limit to max_seq_len tokens

            # Average over heads
            cls_attention = cls_attention_heads.mean(dim=0)  # Shape: (seq_len,)

            # Plot the CLS attention
            plt.figure(figsize=(10, 4))
            plt.bar(range(seq_len), cls_attention.detach().cpu().numpy())
            plt.xticks(range(seq_len), tokens_cleaned, rotation=90, fontsize=8, fontproperties=self.font_prop)
            plt.xlabel('Tokens', fontproperties=self.font_prop)
            plt.ylabel('Attention Weight', fontproperties=self.font_prop)
            plt.title(f'CLS Token Attention to Tokens (Last Layer)\nTest Index: {idx}', fontproperties=self.font_prop)
            plt.tight_layout()
            plt.savefig(os.path.join(self.save_path, f"cls_attention_test_idx_{idx}.png"))
            plt.close()

            # 2. Plot Token-to-Token Attention Heatmap (excluding CLS token)
            # Ignore the first token in the attention matrices
            last_layer_attention = last_layer_attention[:, 1:, 1:]  # Shape: (num_heads, seq_len_full -1, seq_len_full -1)
            last_layer_attention = last_layer_attention[:, :seq_len, :seq_len]  # Limit to max_seq_len tokens

            # Average over heads
            avg_attention = last_layer_attention.mean(dim=0)  # Shape: (seq_len, seq_len)

            # Plot heatmap of attention between tokens
            plt.figure(figsize=(10, 8))
            sns.heatmap(avg_attention.detach().cpu().numpy(), annot=False, cmap='viridis',
                        xticklabels=tokens_cleaned, yticklabels=tokens_cleaned)
            plt.xticks(rotation=90, fontsize=8, fontproperties=self.font_prop)
            plt.yticks(fontsize=8, fontproperties=self.font_prop)
            plt.xlabel('Key Tokens', fontproperties=self.font_prop)
            plt.ylabel('Query Tokens', fontproperties=self.font_prop)
            plt.title(f'Token-to-Token Attention (Averaged over Heads, Last Layer)\nTest Index: {idx}', fontproperties=self.font_prop)
            plt.tight_layout()
            plt.savefig(os.path.join(self.save_path, f"token_attention_heatmap_test_idx_{idx}.png"))
            plt.close()

            # 3. Highlight relevant text based on token importance
            # Compute token importance
            token_importance = avg_attention.sum(dim=0).detach().cpu().numpy()  # Shape: (seq_len,)
