import argparse
from src.data.collect_data import DataCollector
from src.data.curate_data import DataCurator
from src.model.inference import DummyDriver
from src.config import *
import os

def collect(args):
    """Collect driving data from the game"""
    collector = DataCollector(output_dir=RAW_DATA_DIR)
    collector.run()

def curate(args):
    """Review and organize collected data"""
    session_dir = os.path.join(RAW_DATA_DIR, args.data_dir)
    if not os.path.exists(session_dir):
        print(f"Error: Session directory not found: {session_dir}")
        return
    curator = DataCurator(session_dir)
    curator.run()

def preprocess(args):
    """Preprocess collected data for training"""
    print(f"[TODO] Preprocessing data from: {args.data_dir}")
    print("This will prepare the data for training")

def train(args):
    """Train the model on preprocessed data"""
    print(f"[TODO] Training model on data from: {args.data_dir}")
    print("This will train the neural network")

def validate(args):
    """Validate model performance on test data"""
    print(f"[TODO] Validating model: {args.model_path}")
    print(f"Using test data from: {args.data_dir}")

def infer(args):
    """Run inference using PID control"""
    driver = DummyDriver()
    driver.run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Self-driving car pipeline')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Collect command
    subparsers.add_parser('collect', help='Collect driving data')
    
    # Curate command
    curate_parser = subparsers.add_parser('curate', help='Review and organize collected data')
    curate_parser.add_argument('data_dir', help='Directory containing collected data')
    
    # Preprocess command
    preprocess_parser = subparsers.add_parser('preprocess', help='Preprocess collected data')
    preprocess_parser.add_argument('data_dir', help='Directory containing raw data')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument('data_dir', help='Directory containing preprocessed data')
    
    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate model performance')
    validate_parser.add_argument('data_dir', help='Directory containing test data')
    validate_parser.add_argument('model_path', help='Path to trained model')
    
    # Infer command
    subparsers.add_parser('infer', help='Run real-time inference')
    
    args = parser.parse_args()
    
    # Execute command
    if args.command == 'collect':
        collect(args)
    elif args.command == 'curate':
        curate(args)
    elif args.command == 'preprocess':
        preprocess(args)
    elif args.command == 'train':
        train(args)
    elif args.command == 'validate':
        validate(args)
    elif args.command == 'infer':
        infer(args)
    else:
        parser.print_help() 