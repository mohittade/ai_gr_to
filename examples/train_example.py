"""
Example script for training the translation models
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader
from models.transformer import Transformer
from training.train import Trainer, TranslationDataset
from utils.preprocessing import DataPreprocessor, BilingualDataset

def train_german_english_model():
    """
    Train the German→English translation model
    """
    print("=" * 80)
    print("Training German → English Translation Model")
    print("=" * 80)
    
    # Configuration
    config = {
        'd_model': 512,
        'num_heads': 8,
        'num_encoder_layers': 6,
        'num_decoder_layers': 6,
        'd_ff': 2048,
        'dropout': 0.1,
        'max_seq_length': 128,
        'batch_size': 32,
        'num_epochs': 30,
        'learning_rate': 0.0001,
        'warmup_steps': 4000,
        'label_smoothing': 0.1
    }
    
    print("\n[1] Loading and preprocessing data...")
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor(config={
        'max_length': config['max_seq_length'],
        'min_length': 3,
        'lowercase': True
    })
    
    # Load datasets
    # TODO: Replace with actual dataset paths
    train_dataset = BilingualDataset(
        source_file='data/german_english/train.de',
        target_file='data/german_english/train.en',
        preprocessor=preprocessor
    )
    
    val_dataset = BilingualDataset(
        source_file='data/german_english/val.de',
        target_file='data/german_english/val.en',
        preprocessor=preprocessor
    )
    
    print(f"✓ Train samples: {len(train_dataset)}")
    print(f"✓ Validation samples: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        TranslationDataset(train_dataset.source_texts, train_dataset.target_texts),
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        TranslationDataset(val_dataset.source_texts, val_dataset.target_texts),
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    print("\n[2] Initializing model...")
    
    # Create model
    model = Transformer(
        src_vocab_size=32000,  # BPE vocabulary size
        tgt_vocab_size=32000,
        d_model=config['d_model'],
        num_heads=config['num_heads'],
        num_encoder_layers=config['num_encoder_layers'],
        num_decoder_layers=config['num_decoder_layers'],
        d_ff=config['d_ff'],
        max_seq_length=config['max_seq_length'],
        dropout=config['dropout']
    )
    
    # Print model info
    num_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model initialized with {num_params:,} parameters")
    
    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"✓ Using device: {device}")
    model = model.to(device)
    
    print("\n[3] Starting training...")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device
    )
    
    # Train
    trainer.train(num_epochs=config['num_epochs'])
    
    print("\n[4] Saving final model...")
    trainer.save_checkpoint(
        epoch=config['num_epochs'],
        path='checkpoints/de_en_final_model.pt'
    )
    
    print("\n✓ Training complete!")
    print("=" * 80)

def train_english_marathi_model():
    """
    Train the English→Marathi translation model
    """
    print("=" * 80)
    print("Training English → Marathi Translation Model")
    print("=" * 80)
    
    # Similar configuration as German-English
    config = {
        'd_model': 512,
        'num_heads': 8,
        'num_encoder_layers': 6,
        'num_decoder_layers': 6,
        'd_ff': 2048,
        'dropout': 0.1,
        'max_seq_length': 128,
        'batch_size': 32,
        'num_epochs': 30,
        'learning_rate': 0.0001,
        'warmup_steps': 4000,
        'label_smoothing': 0.1
    }
    
    print("\n[1] Loading and preprocessing data...")
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor(config={
        'max_length': config['max_seq_length'],
        'min_length': 3,
        'lowercase': False  # Keep Devanagari script as-is
    })
    
    # Load datasets
    # TODO: Replace with actual dataset paths
    train_dataset = BilingualDataset(
        source_file='data/english_marathi/train.en',
        target_file='data/english_marathi/train.mr',
        preprocessor=preprocessor
    )
    
    val_dataset = BilingualDataset(
        source_file='data/english_marathi/val.en',
        target_file='data/english_marathi/val.mr',
        preprocessor=preprocessor
    )
    
    print(f"✓ Train samples: {len(train_dataset)}")
    print(f"✓ Validation samples: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        TranslationDataset(train_dataset.source_texts, train_dataset.target_texts),
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        TranslationDataset(val_dataset.source_texts, val_dataset.target_texts),
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    print("\n[2] Initializing model...")
    
    # Create model
    model = Transformer(
        src_vocab_size=32000,
        tgt_vocab_size=32000,
        d_model=config['d_model'],
        num_heads=config['num_heads'],
        num_encoder_layers=config['num_encoder_layers'],
        num_decoder_layers=config['num_decoder_layers'],
        d_ff=config['d_ff'],
        max_seq_length=config['max_seq_length'],
        dropout=config['dropout']
    )
    
    # Print model info
    num_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model initialized with {num_params:,} parameters")
    
    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"✓ Using device: {device}")
    model = model.to(device)
    
    print("\n[3] Starting training...")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device
    )
    
    # Train
    trainer.train(num_epochs=config['num_epochs'])
    
    print("\n[4] Saving final model...")
    trainer.save_checkpoint(
        epoch=config['num_epochs'],
        path='checkpoints/en_mr_final_model.pt'
    )
    
    print("\n✓ Training complete!")
    print("=" * 80)

def main():
    """
    Main training pipeline
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Train translation models')
    parser.add_argument(
        '--model',
        choices=['de-en', 'en-mr', 'both'],
        default='both',
        help='Which model to train'
    )
    
    args = parser.parse_args()
    
    # Create checkpoints directory
    os.makedirs('checkpoints', exist_ok=True)
    
    if args.model in ['de-en', 'both']:
        train_german_english_model()
        print("\n")
    
    if args.model in ['en-mr', 'both']:
        train_english_marathi_model()
    
    print("\n🎉 All training complete!")

if __name__ == "__main__":
    main()
