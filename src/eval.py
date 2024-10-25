from utils.evaluate_utils import ModelEvaluator
from utils.data_utils import preprocess_tir_news, preprocess_eval_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from models.baseline import NewsClassificationModel
import argparse
from peft import LoraConfig, TaskType, get_peft_model
from peft.peft_model import PeftModel, PeftModelForSequenceClassification


parser = argparse.ArgumentParser()
parser.add_argument('--model_run', default='baseline', help="One of ['baseline', 'first-finetuning', 'contrastive']")
parser.add_argument('--epochs', type=int, default=15, help='Number of epochs to train for')
parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
parser.add_argument('--save_path', default='./checkpoints/baseline/', help='Directory to save checkpoints to')
parser.add_argument('--learning-rate',type=float, default=.0005, help='Learning rate')
parser.add_argument('--checkpoint-path', default=None, help=('Checkpoint path to load from'))
parser.add_argument('--model_name',  default="fgaim/tielectra-small", help=('Checkpoint path to load from'))
parser.add_argument('--run_name',  default="baseline", help=('Checkpoint path to load from'))
parser.add_argument('--lora_checkpoint',  default=False, help=('Check if LoRA checkpoint'))


if __name__ == "__main__":

    args = parser.parse_args()
    train_dataset, val_dataset, test_dataset = preprocess_tir_news()
    model = NewsClassificationModel(args.checkpoint_path, args.checkpoint_path, train_dataset, val_dataset, test_dataset, use_lora=False, 
                                    checkpoint_path=args.checkpoint_path, lora_checkpoint=args.lora_checkpoint)
    

# # Model and tokenizer
#     tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_path)
#     tokenizer.model_max_length = 512

#     peft_config = LoraConfig.from_pretrained(args.checkpoint_path)
#     model = AutoModelForSequenceClassification.from_pretrained(
#             args.model_name, 
#             num_labels=2,
#             ignore_mismatched_sizes=True
#         )
    
#     test_dataset = tokenize_text(test_dataset, tokenizer)

#     #model = PeftModel.from_pretrained(model, "./checkpoints/contrastive")
#     model = PeftModel.from_pretrained(model, args.checkpoint_path)

    model_eval = ModelEvaluator(model.model, save_path=args.save_path)
    model_eval.evaluate_classification(model.test_dataset)
    train_dataset, val_dataset, test_dataset = preprocess_eval_dataset("./data/amh.txt", 
                                                                        "./data/tir.txt", 
                                                                        "./data/eng.txt",
                                                                         "./data/metadata.tsv",
                                                                        model.tokenizer, 
                                                                        label_name='Category')
    model_eval.visualize_attention(test_dataset, [i for i in range(14)], model.tokenizer)
    model_eval.highlight_text(model.test_dataset.filter(lambda x: x['label'] == 1), [i for i in range(14)], model.tokenizer)


    
