import argparse
from pathlib import Path
from src.data.collect_data import DataCollector
from src.data.curate_data import DataCurator
from src.data.preprocess_data import process_session
from src.model.cnn.train import train
from src.model.inference import inference

def main():
    parser = argparse.ArgumentParser(description='Self-driving car data pipeline')
    parser.add_argument('action', choices=['collect', 'curate', 'preprocess', 'train', 'drive'],
                      help='Action to perform')
    
    # Only required for curation and preprocessing
    parser.add_argument('--session', help='Session directory to curate/preprocess')
    
    # Optional for drive (if not provided, uses random agent)
    parser.add_argument('--model', help='Path to trained model')
    
    args = parser.parse_args()
    
    if args.action == 'collect':
        collector = DataCollector()
        collector.run()
    
    elif args.action == 'curate':
        if not args.session:
            parser.error('curate requires --session')
        curator = DataCurator(args.session)
        curator.run()
    
    elif args.action == 'preprocess':
        if not args.session:
            parser.error('preprocess requires --session')
        process_session(args.session)
    
    elif args.action == 'train':
        train()
    
    elif args.action == 'drive':
        inference(args.model)  # If model is None, will use random agent

if __name__ == '__main__':
    main() 