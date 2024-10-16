from src.data_utils import preprocess_amharic_news, tokenize_text, preprocess_distil_dataset
from src.baseline import NewsClassificationModel
from transformers import AutoTokenizer
from src.distillation import DistillationNet

if __name__ == "__main__":
    train_dataset, val_dataset, test_dataset = preprocess_amharic_news()

    model = NewsClassificationModel(model_name="fgaim/tielectra-small",
                                    tokenizer_name="fgaim/tielectra-small",
                                    train_dataset=train_dataset,
                                    val_dataset=val_dataset,
                                    test_dataset=test_dataset, checkpoint_path="./distillation")

    model.train(batch_size=10, num_epochs=30)
    model.evaluate([0])

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

