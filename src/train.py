from utils.data_utils import preprocess_tir_news, tokenize_text, preprocess_distil_dataset, preprocess_contrastive_dataset
from models.baseline import NewsClassificationModel
from transformers import AutoTokenizer
from models.contrastive_learning import ContrastiveNet
import argparse
import pandas as pd

# Extract args for running baseline or the fine-tunings
parser = argparse.ArgumentParser()
parser.add_argument('--baseline', default=True, help='Run baseline model train or fine tuning with contrastive learning')
parser.add_argument('--epochs', type=int, default=15, help='Number of epochs to train for')
parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
parser.add_argument('--save_path', default='./checkpoints/baseline/', help='Directory to save checkpoints to')
parser.add_argument('--learning-rate',type=float, default=.0005, help='Learning rate')
parser.add_argument('--checkpoint-path',  type=float, default=None, help=('Checkpoint path to load from'))
parser.add_argument('--model_name',  type=float, default="fgaim/tielectra-small", help=('Checkpoint path to load from'))

def train_baseline(args):
    
    train_dataset, val_dataset, test_dataset = preprocess_tir_news()

    model = NewsClassificationModel(model_name= args.model_name,
                                    tokenizer_name= args.model_name,
                                    train_dataset=train_dataset,
                                    val_dataset=val_dataset,
                                    test_dataset=test_dataset, 
                                    checkpoint_path= args.checkpoint_path,
                                    save_path=args.save_path)

    model.train(batch_size=args.batch_size, num_epochs=args.epochs)

def train_contrastive_learning(args):

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_dataset, val_dataset, test_dataset = preprocess_contrastive_dataset("./data/amh.txt", 
                                                                              "./data/tir.txt", 
                                                                              "./data/metadata.tsv", 
                                                                              tokenizer, label_name='Category')
    
    
    contrastive_model = ContrastiveNet(
        model_name=args.model_name,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        batch_size=args.batch_size
    )

    # Train
    contrastive_model.train_contrastive(lr=args.learning_rate, n_epochs=args.epochs)


if __name__ == "__main__":
    args = parser.parse_args()
    if args.baseline:
        train_baseline(args)
    else:
        train_contrastive_learning(args)

    
    

    