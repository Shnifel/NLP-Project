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


class DistillationNet:

    def __init__(self, student_name=None, teacher_name=None,
                 train_dataset=None, val_dataset=None, test_dataset=None, batch_size = 32):
        
        self.student = NewsClassificationModel(student_name, student_name, train_dataset, val_dataset, test_dataset, tokenize=False, n_labels=5)
        self.teacher = NewsClassificationModel(teacher_name, teacher_name, train_dataset, val_dataset, test_dataset, 
                                               tokenize=False, n_labels=5, use_lora=False)

        self.student_model = self.student.model
        self.teacher_model = self.teacher.model

        self.train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)

    def distill_student(self, lr, n_epochs, temp, alpha):
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.teacher_model.to(device)
        self.student_model.to(device)
        self.teacher_model.eval()

        # Optimizer and scheduler
        optimizer = torch.optim.AdamW(self.student_model.parameters(), lr=lr)
        num_training_steps = n_epochs * len(self.train_dataloader)
        lr_scheduler = get_scheduler(
            "linear",
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=num_training_steps
        )

        # Loss functions
        ce_loss = nn.CrossEntropyLoss()
        kl_loss = nn.KLDivLoss(reduction='batchmean')

        # Training loop
        progress_bar = tqdm(range(num_training_steps))

        self.student_model.train()
        for epoch in range(n_epochs):
            for batch in self.train_dataloader:
                # Move batch to device

                batch = {k: v.to(device) for k, v in batch.items()}
                  
                # Student forward pass on Tigrinya inputs
                student_outputs = self.student_model(
                    input_ids=batch['tir_input_ids'],
                    attention_mask=batch['tir_attention_mask']
                )
                student_logits = student_outputs.logits

                # Teacher forward pass on English inputs (no gradient)
                with torch.no_grad():
                    teacher_outputs = self.teacher_model(
                        input_ids=batch['eng_input_ids'],
                        attention_mask=batch['eng_attention_mask']
                    )
                    teacher_logits = teacher_outputs.logits

                # Compute losses
                student_logits = student_logits.float()
                labels = batch['label'].long()
                loss_ce = ce_loss(student_logits, labels)
                loss_kl = kl_loss(
                    nn.functional.log_softmax(student_logits / temp, dim=-1),
                    nn.functional.softmax(teacher_logits / temp, dim=-1)
                ) * (temp ** 2)
                loss = alpha * loss_ce + (1 - alpha) * loss_kl

                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                progress_bar.update(1)

            print(f"Epoch {epoch}, Loss: {loss.item()}")

            self.student_model.save_pretrained('./distillation')
            

    def evaluate(self):
        # Evaluation
        self.student_model.eval()

        # for batch in self.val_dataloader:
        #     batch = {k: v.to(device) for k, v in batch.items()}
        #     with torch.no_grad():
        #         outputs = self.student_model(
        #             input_ids=batch['tigrinya_input_ids'],
        #             attention_mask=batch['tigrinya_attention_mask']
        #         )
        #     predictions = torch.argmax(outputs.logits, dim=-1)
        #     metric.add_batch(predictions=predictions, references=batch['labels'])

        # eval_result = metric.compute()
        # print(eval_result)


                



