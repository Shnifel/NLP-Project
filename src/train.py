from utils.data_utils import preprocess_tir_news, tokenize_text, preprocess_amh_news, preprocess_contrastive_dataset, preprocess_eval_dataset
from models.baseline import NewsClassificationModel
from models.masked_lm import MaskedLMModel
from transformers import AutoTokenizer
from models.contrastive_learning import ContrastiveNet
import argparse
import pandas as pd
from eval import ModelEvaluator

# Extract args for running baseline or the fine-tunings
parser = argparse.ArgumentParser()
parser.add_argument('--model_run', default='baseline', help="One of ['baseline', 'first-finetuning', 'contrastive']")
parser.add_argument('--epochs', type=int, default=15, help='Number of epochs to train for')
parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
parser.add_argument('--save_path', default='./checkpoints/baseline/', help='Directory to save checkpoints to')
parser.add_argument('--learning-rate',type=float, default=.0005, help='Learning rate')
parser.add_argument('--checkpoint-path', default=None, help=('Checkpoint path to load from'))
parser.add_argument('--model_name',  default="fgaim/tielectra-small", help=('Checkpoint path to load from'))
parser.add_argument('--run_name',  default="baseline", help=('Checkpoint path to load from'))

def train_baseline(args):
    
    train_dataset, val_dataset, test_dataset = preprocess_tir_news()

    model = NewsClassificationModel(model_name= args.model_name,
                                    tokenizer_name= args.model_name,
                                    train_dataset=train_dataset,
                                    val_dataset=val_dataset,
                                    test_dataset=test_dataset, 
                                    checkpoint_path= args.checkpoint_path,
                                    save_path=args.save_path,
                                    lora_checkpoint=False
                                    )

    model.train(batch_size=args.batch_size, num_epochs=args.epochs, run_name=args.run_name)

    return model

def train_amharic_finetuning(args):
    
    train_dataset, val_dataset, test_dataset = preprocess_amh_news()

    model = MaskedLMModel(model_name= args.model_name,
                                    tokenizer_name= args.model_name,
                                    train_dataset=train_dataset,
                                    val_dataset=val_dataset,
                                    test_dataset=test_dataset, 
                                    checkpoint_path= args.checkpoint_path,
                                    save_path=args.save_path, 
                                    use_lora=False
                                    )

    model.train(batch_size=args.batch_size, num_epochs=args.epochs,run_name=args.run_name)
    return model

def train_contrastive_learning(args):

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.model_max_length = 64
    train_dataset, val_dataset, test_dataset = preprocess_contrastive_dataset("./data/amh.txt", 
                                                                              "./data/tir.txt", 
                                                                              "./data/metadata.tsv", 
             
                                                                     tokenizer, label_name='Category')
    print(train_dataset[0])
    print(tokenizer.convert_ids_to_tokens(train_dataset[0]['tir_positive_input_ids']))
    tokenizer.save_pretrained(args.save_path)
    
    
    contrastive_model = ContrastiveNet(
        model_name=args.model_name,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        batch_size=args.batch_size,
        checkpoint_path=args.checkpoint_path
    )

    contrastive_model.plot_tsne("after")

    # Train
    # contrastive_model.train_contrastive(lr=args.learning_rate, n_epochs=args.epochs, save_path=args.save_path, run_name=args.run_name)

    # contrastive_model.plot_tsne("after")

if __name__ == "__main__":
    args = parser.parse_args()
    if args.model_run == 'baseline':
        model = train_baseline(args)
        model_eval = ModelEvaluator(model.model, save_path=f'./results/{args.run_name}')
        model_eval.evaluate_classification(model.test_dataset)
        train_dataset, val_dataset, test_dataset = preprocess_eval_dataset("./data/amh.txt", 
                                                                        "./data/tir.txt", 
                                                                        "./data/eng.txt",
                                                                         "./data/metadata.tsv",
                                                                        model.tokenizer, 
                                                                        label_name='Category')
        model_eval.visualize_attention(test_dataset, [i for i in range(14)], model.tokenizer)
        model_eval.highlight_text(model.test_dataset.filter(lambda x: x['label'] == 1), [i for i in range(14)], model.tokenizer)
    elif args.model_run == 'first-finetuning':
        train_amharic_finetuning(args)
    else:
        train_contrastive_learning(args)

    
    

    