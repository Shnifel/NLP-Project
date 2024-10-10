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

class NewsClassificationModel:
    def __init__(self, model_name, tokenizer_name, train_dataset, val_dataset, test_dataset, 
                 use_lora = True, save_path = "./baseline", checkpoint_path = None):

        # Model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.tokenizer.model_max_length = 512

        if checkpoint_path is not None:
            # Load the LoRA configuration
            peft_config = LoraConfig.from_pretrained(checkpoint_path)
            # Load the base model
            self.model = AutoModelForSequenceClassification.from_pretrained(
                peft_config.base_model_name_or_path, 
                num_labels=2
            )

            print("HERE")
            # Load the model with LoRA weights
            self.model = PeftModel.from_pretrained(self.model, checkpoint_path)
        else:
            # Load the base model
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
            # Apply LoRA if specified
            if use_lora:
                config = LoraConfig(
                    task_type=TaskType.SEQ_CLS,
                    r=8,
                    lora_alpha=32,
                    target_modules=["query", "value"],
                    lora_dropout=0.01
                )
                self.model = get_peft_model(self.model, config, "default")
        
        # Tokenize text
        self.train_dataset = tokenize_text(train_dataset, self.tokenizer)
        self.val_dataset = tokenize_text(val_dataset, self.tokenizer)
        self.test_dataset = tokenize_text(test_dataset, self.tokenizer)
        
        self.save_path = save_path

    def train(self, batch_size = 32, num_epochs = 3):
        training_args = TrainingArguments(
            output_dir= self.save_path,
            num_train_epochs= num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=100,
            weight_decay=0.01,
            logging_dir='./logs',
            evaluation_strategy="epoch",
        )
        trainer = Trainer(
            model= self.model,
            args=training_args,
            train_dataset= self.train_dataset,
            eval_dataset= self.val_dataset,
            compute_metrics=compute_metrics
        )

        trainer.train()

    def evaluate(self, selected_indices):
        import os
        import numpy as np
        import torch
        import matplotlib.pyplot as plt
        import seaborn as sns
        from transformers import Trainer
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, cohen_kappa_score
        import matplotlib.font_manager as font_manager
        import matplotlib

        # Ensure save path exists
        os.makedirs(self.save_path, exist_ok=True)

        # Configure matplotlib to use a font that supports Amharic characters
        # Replace '/path/to/NotoSansEthiopic-Regular.ttf' with the actual path to the font file
        font_path = "C:/Users/Muhammad Sahal/Downloads/Noto_Sans_Ethiopic/static/NotoSansEthiopic-Regular.ttf"  # Update this path
        font_prop = font_manager.FontProperties(fname=font_path)
        matplotlib.rcParams['font.family'] = font_prop.get_name()

        # Create trainer for evaluation
        trainer = Trainer(
            model=self.model,
        )

        # Get predictions
        prediction_output = trainer.predict(self.test_dataset)
        predictions = prediction_output.predictions
        label_ids = prediction_output.label_ids

        # Process predictions
        preds = np.argmax(predictions, axis=1)

        # Compute metrics
        accuracy = accuracy_score(label_ids, preds)
        precision = precision_score(label_ids, preds, average='binary')
        recall = recall_score(label_ids, preds, average='binary')
        f1 = f1_score(label_ids, preds, average='binary')
        kappa = cohen_kappa_score(label_ids, preds)

        # Save metrics
        with open(os.path.join(self.save_path, "evaluation_metrics.txt"), "w", encoding='utf-8') as f:
            f.write(f"Accuracy: {accuracy}\n")
            f.write(f"Precision: {precision}\n")
            f.write(f"Recall: {recall}\n")
            f.write(f"F1-score: {f1}\n")
            f.write(f"Cohen's Kappa: {kappa}\n")

        # Compute confusion matrix
        cm = confusion_matrix(label_ids, preds)

        # Save confusion matrix
        plt.figure(figsize=(8,6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title('Confusion Matrix')
        plt.savefig(os.path.join(self.save_path, "confusion_matrix.png"))
        plt.close()

        # For selected indices, visualize attention and highlight relevant text
        for idx in selected_indices:
            # Get input features
            sample = self.test_dataset[idx]

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
            plt.xticks(range(seq_len), tokens_cleaned, rotation=90, fontsize=8, fontproperties=font_prop)
            plt.xlabel('Tokens', fontproperties=font_prop)
            plt.ylabel('Attention Weight', fontproperties=font_prop)
            plt.title(f'CLS Token Attention to Tokens (Last Layer)\nTest Index: {idx}', fontproperties=font_prop)
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
            plt.xticks(rotation=90, fontsize=8, fontproperties=font_prop)
            plt.yticks(fontsize=8, fontproperties=font_prop)
            plt.xlabel('Key Tokens', fontproperties=font_prop)
            plt.ylabel('Query Tokens', fontproperties=font_prop)
            plt.title(f'Token-to-Token Attention (Averaged over Heads, Last Layer)\nTest Index: {idx}', fontproperties=font_prop)
            plt.tight_layout()
            plt.savefig(os.path.join(self.save_path, f"token_attention_heatmap_test_idx_{idx}.png"))
            plt.close()

            # 3. Highlight relevant text based on token importance
            # Compute token importance
            token_importance = avg_attention.sum(dim=0).detach().cpu().numpy()  # Shape: (seq_len,)

            # Normalize importance scores
            importance_range = token_importance.max() - token_importance.min()
            if importance_range == 0:
                importance_scores = np.zeros_like(token_importance)
            else:
                importance_scores = (token_importance - token_importance.min()) / importance_range

            # Create HTML text with highlighted tokens
            html_text = ""
            for tok, score in zip(tokens_cleaned, importance_scores):
                color_intensity = int(255 - score * 255)
                color_hex = f'#ff{color_intensity:02x}{color_intensity:02x}'  # From white to red
                html_text += f'<span style="background-color:{color_hex}; text-decoration: none;">{tok} </span>'

            # Wrap the content in basic HTML structure to prevent external styles from interfering
            html_content = f"""
            <html>
            <head>
            <meta charset="UTF-8">
            <style>
            span {{
                text-decoration: none;
                font-family: '{font_prop.get_name()}';
            }}
            </style>
            </head>
            <body>
            <p>{html_text}</p>
            </body>
            </html>
            """

            # Save HTML to file
            html_file = os.path.join(self.save_path, f"highlighted_text_test_idx_{idx}.html")
            with open(html_file, "w", encoding='utf-8') as f:
                f.write(html_content)

            # Convert HTML to PDF (requires pdfkit and wkhtmltopdf)
            try:
                import pdfkit
                pdf_file = os.path.join(self.save_path, f"highlighted_text_test_idx_{idx}.pdf")
                pdfkit.from_file(html_file, pdf_file)
            except Exception as e:
                print(f"Error converting HTML to PDF for test index {idx}: {e}")
                print("Please ensure that pdfkit and wkhtmltopdf are installed.")
