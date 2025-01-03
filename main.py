import argparse
from pathlib import Path
from src.data.collect_data import collect_data
from src.data.curate_data import curate_data
from src.data.preprocess_data import process_session
from src.model.cnn.train import train
from src.model.inference import inference

def main():
    parser = argparse.ArgumentParser(description='Self-driving car data pipeline')
    parser.add_argument('action', choices=['collect', 'curate', 'preprocess', 'train', 'drive'],
                      help='Action to perform')
    
    # Only required for curation and preprocessing
    parser.add_argument('--session', help='Session directory to curate/preprocess')
    
    # Only required for inference
    parser.add_argument('--model', help='Path to trained model')
    
    args = parser.parse_args()
    
    if args.action == 'collect':
        collect_data()
    
    elif args.action == 'curate':
        if not args.session:
            parser.error('curate requires --session')
        curate_data(args.session)
    
    elif args.action == 'preprocess':
        if not args.session:
            parser.error('preprocess requires --session')
        process_session(args.session)
    
    elif args.action == 'train':
        train()
    
    elif args.action == 'drive':
        if not args.model:
            parser.error('drive requires --model')
        inference(args.model)

if __name__ == '__main__':
    main() 