from src.data_utils import preprocess_amharic_news, tokenize_text
from src.baseline import NewsClassificationModel

if __name__ == "__main__":
    train_dataset, val_dataset, test_dataset = preprocess_amharic_news()

    model = NewsClassificationModel(model_name="fgaim/tielectra-small",
                                    tokenizer_name="fgaim/tielectra-small",
                                    train_dataset=train_dataset,
                                    val_dataset=val_dataset,
                                    test_dataset=test_dataset, checkpoint_path="./baseline/checkpoint-1560")

    model.evaluate([0])