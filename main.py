from src.data_utils import preprocess_amharic_news, tokenize_text, preprocess_distil_dataset, preprocess_contrastive_dataset
from src.baseline import NewsClassificationModel
from transformers import AutoTokenizer
from src.distillation import DistillationNet
from src.contrastive_learning import ContrastiveNet
import pandas as pd

if __name__ == "__main__":
    # -----------------------------------Baseline----------------------------
    # train_dataset, val_dataset, test_dataset = preprocess_amharic_news()

    # model = NewsClassificationModel(model_name="fgaim/tielectra-small",
    #                                 tokenizer_name="fgaim/tielectra-small",
    #                                 train_dataset=train_dataset,
    #                                 val_dataset=val_dataset,
    #                                 test_dataset=test_dataset, checkpoint_path="./distillation") #checkpoint from distillation models

    # model.train(batch_size=10, num_epochs=30)
    # model.evaluate([0])
    
    
    # -------------------Distillation

    # student_model_name = "fgaim/tielectra-small"
    # student_tokenizer = AutoTokenizer.from_pretrained(student_model_name)
    # student_tokenizer.model_max_length = 512

    # teacher_model_name = "AyoubChLin/Albert-bbc-news"
    # teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_model_name)
    # teacher_tokenizer.model_max_length = 512

    # train, val, test = preprocess_distil_dataset("./data/eng.txt", "./data/tir.txt", "./data/metadata.tsv", 
    #                                              student_tokenizer, teacher_tokenizer)
    
    # distil_net = DistillationNet(student_model_name, teacher_model_name, train, val, test)
    # distil_net.distill_student(5e-3, 3, 2., 0.5)


    # -------------------contrastive
    model_name = "fgaim/tielectra-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_dataset, val_dataset, test_dataset = preprocess_contrastive_dataset("./data/amh.txt", "./data/tir.txt", "./data/metadata.tsv", tokenizer, label_name='Category')
    
    for i in range(5):
        print(f"Sample {i + 1}:")
        print(f"Anchor Input IDs: {train_dataset[i]['tir_anchor_input_ids']}")
        # print(f"Anchor Attention Mask: {train_dataset[i]['tir_anchor_attention_mask']}")
        print(f"Positive Input IDs: {train_dataset[i]['tir_positive_input_ids']}")
        # print(f"Positive Attention Mask: {train_dataset[i]['tir_positive_attention_mask']}")
        print(f"Label: {train_dataset[i]['label']}")
        print("-" * 50)

    
    contrastive_model = ContrastiveNet(
        model_name=model_name,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        batch_size=8
    )

    # Train
    train_losses = contrastive_model.train_contrastive(lr=2e-5, n_epochs=3)

    # Evaluate the model
    eval_results = contrastive_model.evaluate()
    print(f"Validation Loss: {eval_results['validation_loss']:.4f}")
