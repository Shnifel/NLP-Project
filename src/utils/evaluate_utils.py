from IPython.display import clear_output
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import Trainer
from sklearn.metrics import confusion_matrix, cohen_kappa_score
import matplotlib.font_manager as font_manager
from transformers import Trainer
import matplotlib
import evaluate
import numpy as np
from utils.attention_utils import display_CLS_layers, display_token_to_token
from utils.highlight_text import highlight_relevant_text

# Load metrics
accuracy_metric = evaluate.load("accuracy")
precision_metric = evaluate.load("precision")
recall_metric = evaluate.load("recall")
f1_metric = evaluate.load("f1")

def compute_metrics(p): 
    preds = np.argmax(p.predictions, axis=1)
    labels = p.label_ids
    
    # Compute accuracy, precision, recall, and F1
    accuracy = accuracy_metric.compute(predictions=preds, references=labels)
    precision = precision_metric.compute(predictions=preds, references=labels, average="binary")
    recall = recall_metric.compute(predictions=preds, references=labels, average="binary")
    f1 = f1_metric.compute(predictions=preds, references=labels, average="binary")
    cohens_kappa = cohen_kappa_score(preds, labels)
    
    return {
        'accuracy': accuracy['accuracy'],
        'precision': precision['precision'],
        'recall': recall['recall'],
        'f1': f1['f1'],
        'cohens_kappa': cohens_kappa
    }

def compute_metrics_multiclass(p):
    preds = np.argmax(p.predictions, axis=1)
    labels = p.label_ids
    
    # Compute accuracy, precision, recall, and F1
    accuracy = accuracy_metric.compute(predictions=preds, references=labels)
    precision = precision_metric.compute(predictions=preds, references=labels, average="macro")
    recall = recall_metric.compute(predictions=preds, references=labels, average="macro")
    f1 = f1_metric.compute(predictions=preds, references=labels, average="macro")
    cohens_kappa = cohen_kappa_score(preds, labels)
    
    return {
        'accuracy': accuracy['accuracy'],
        'precision': precision['precision'],
        'recall': recall['recall'],
        'f1': f1['f1'],
        'cohens_kappa': cohens_kappa
    }


class ModelEvaluator:

    def __init__(self, model, save_path):

        self.model = model # Model to evaluate
        self.save_path = save_path # Path to save results to
        os.makedirs(self.save_path, exist_ok=True)
        os.makedirs(os.path.join(self.save_path, "attention_vis"), exist_ok=True)
        os.makedirs(os.path.join(self.save_path, "text_highlighted"), exist_ok=True)

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
            f.write(str(metrics))
        
        # Compute confusion matrix
        cm = confusion_matrix(preds.label_ids, np.argmax(preds.predictions, axis = 1))
        
        # Save confusion matrix
        plt.figure(figsize=(8,6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title('Confusion Matrix')
        plt.savefig(os.path.join(self.save_path, "confusion_matrix.pdf"))
        plt.close()

    def visualize_attention(self, test_dataset, selected_indices, tokenizer):
        for idx in selected_indices:

            article = test_dataset[idx]
            input_ids = torch.tensor(article['tir_input_ids'], dtype=torch.long).unsqueeze(0).to(self.model.device)
            attention_mask = torch.tensor(article['tir_attention_mask'], dtype=torch.long).unsqueeze(0).to(self.model.device)

            inputs = {'input_ids': input_ids, 'attention_mask': attention_mask}

            with torch.no_grad():
                outputs = self.model(**inputs, output_attentions=True)

            attentions = outputs.attentions  # Tuple of (num_layers, batch_size, num_heads, seq_len, seq_len)

            tokens = tokenizer.convert_ids_to_tokens(article['tir_input_ids'])

            max_seq_len = 100
            seq_len = min(len(tokens), max_seq_len, attention_mask.sum()) # excludes padding tokens
            tokens = tokens[:seq_len]

            display_CLS_layers(attentions, tokens, self.font_prop, 
                               save_path=os.path.join(self.save_path, f"attention_vis/cls_vs_layers_{idx}.pdf"), 
                               eng_trans = article['eng'] ,
                               seq_len= seq_len)
            
            display_token_to_token(attentions, tokens, self.font_prop, 
                    save_path=os.path.join(self.save_path, f"attention_vis/tokens_vs_tokens_{idx}.pdf"), 
                    eng_trans = None ,
                    seq_len= seq_len)


    def highlight_text(self, test_dataset, selected_indices, tokenizer):

        for idx in selected_indices:

            article = test_dataset[idx]
            input_ids = torch.tensor(article['input_ids'], dtype=torch.long).unsqueeze(0).to(self.model.device)
            attention_mask = torch.tensor(article['attention_mask'], dtype=torch.long).unsqueeze(0).to(self.model.device)

            inputs = {'input_ids': input_ids, 'attention_mask': attention_mask}

            with torch.no_grad():
                outputs = self.model(**inputs, output_attentions=True)

            attentions = outputs.attentions  # Tuple of (num_layers, batch_size, num_heads, seq_len, seq_len)

            tokens = tokenizer.convert_ids_to_tokens(article['input_ids'])

            seq_len = attention_mask.sum() # excludes padding tokens

            highlight_relevant_text(tokens, attentions, os.path.join(self.save_path, "text_highlighted"), 
                                    self.font_prop, seq_len, article['text'], index=idx)