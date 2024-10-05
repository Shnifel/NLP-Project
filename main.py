from src.data_utils import construct_data_loaders, tokenize_text
from src.train import train_model
from IPython.display import clear_output
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from peft import LoraModel, LoraConfig, TaskType, get_peft_model

if __name__ == "__main__":
    train_dataset, test_dataset = construct_data_loaders("data/SwahiliNewsClassificationDataset.csv")

    tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
    train_dataset = tokenize_text(train_dataset, tokenizer)
    test_dataset = tokenize_text(test_dataset, tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained("bert-base-multilingual-cased", num_labels=2)

    config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=8,
        lora_alpha=32,
        target_modules=["query", "value"],
        lora_dropout=0.01
    )

    model = get_peft_model(model, config, "default")

    train_model(model, train_dataset, test_dataset, 10)