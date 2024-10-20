from utils.data_utils import tokenize_text
from transformers import AutoTokenizer, AutoModel
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments, EvalPrediction
from peft import LoraConfig, TaskType, get_peft_model
from peft.peft_model import PeftModel, PeftModelForSequenceClassification
from utils.evaluate_utils import compute_metrics, compute_metrics_multiclass
import matplotlib.pyplot as plt
import numpy as np

class NewsClassificationModel:
    def __init__(self, model_name=None, tokenizer_name=None, train_dataset=None, val_dataset=None, test_dataset=None, 
                 use_lora=True, save_path="./baseline", checkpoint_path=None, lora_checkpoint = True, tokenize=True, n_labels=2, is_contrastive=False):
        
        self.n_labels = n_labels

        # Model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.tokenizer.model_max_length = 512

        if checkpoint_path is not None:
            if lora_checkpoint:
                # Load the LoRA configuration
                peft_config = LoraConfig.from_pretrained(checkpoint_path)
                # Load the base model
                if is_contrastive:
                    self.model = AutoModel.from_pretrained(
                        peft_config.base_model_name_or_path,
                        ignore_mismatched_sizes=True
                    )
                else:
                    self.model = AutoModelForSequenceClassification.from_pretrained(
                        peft_config.base_model_name_or_path, 
                        num_labels=n_labels,
                        ignore_mismatched_sizes=True
                    )

                print("Loading checkpoint")
                # Load the model with LoRA weights
                self.model = PeftModel.from_pretrained(self.model, checkpoint_path)
            else:
                self.model = AutoModelForSequenceClassification.from_pretrained(
                        checkpoint_path, 
                        num_labels=n_labels,
                        ignore_mismatched_sizes=True
                    )

        else:
            # Load the base model
            if is_contrastive:
                self.model = AutoModel.from_pretrained(
                    model_name,
                    ignore_mismatched_sizes=True
                )
            else:
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    model_name, 
                    num_labels=n_labels,
                    ignore_mismatched_sizes=True
                )

        # Apply LoRA if specified
        if use_lora:
            config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION if is_contrastive else TaskType.SEQ_CLS,
                r=8,
                auto_mapping='auto',
                lora_alpha=32,
                target_modules=["query", "value"],
                lora_dropout=0.01
            )
            self.model = get_peft_model(self.model, config)
        
        # Tokenize text
        if tokenize:
            self.train_dataset = tokenize_text(train_dataset, self.tokenizer)
            self.val_dataset = tokenize_text(val_dataset, self.tokenizer)
            self.test_dataset = tokenize_text(test_dataset, self.tokenizer)
        else:
            self.train_dataset = train_dataset
            self.val_dataset = val_dataset
            self.test_dataset = test_dataset
        
        self.save_path = save_path

    def train(self, batch_size = 32, num_epochs = 3, run_name = 'baseline'):
        if self.n_labels == 2:
            comp_metrics = compute_metrics
        else:
            comp_metrics = compute_metrics_multiclass
        print(self.train_dataset[0])
        training_args = TrainingArguments(
            output_dir= self.save_path,
            num_train_epochs= num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=100,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps = 30,
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            save_total_limit=1,
            evaluation_strategy="epoch",
            run_name=run_name
        )
        trainer = Trainer(
            model= self.model,
            args=training_args,
            train_dataset= self.train_dataset,
            eval_dataset= self.val_dataset,
            tokenizer=self.tokenizer,
            compute_metrics=comp_metrics
        )

        trainer.train()

    def evaluate(self, selected_indices):
       
        
        pass

        



     