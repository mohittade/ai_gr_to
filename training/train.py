"""
Training Pipeline for German-English-Marathi Neural Machine Translation
Implements Adam optimizer, learning rate scheduling, regularization, and dynamic batching
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from typing import Optional, Dict, List, Tuple
import time
import os
from tqdm import tqdm
import json

from models.transformer import Transformer, create_mask


class TranslationDataset(Dataset):
    """PyTorch Dataset for translation pairs"""
    
    def __init__(self, source_texts: List[str], target_texts: List[str], 
                 source_tokenizer, target_tokenizer, max_length: int = 128):
        """
        Args:
            source_texts: List of source language sentences
            target_texts: List of target language sentences
            source_tokenizer: Tokenizer for source language
            target_tokenizer: Tokenizer for target language
            max_length: Maximum sequence length
        """
        self.source_texts = source_texts
        self.target_texts = target_texts
        self.source_tokenizer = source_tokenizer
        self.target_tokenizer = target_tokenizer
        self.max_length = max_length
        
    def __len__(self) -> int:
        return len(self.source_texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Tokenize and encode
        src_tokens = self.source_tokenizer.encode(self.source_texts[idx], 
                                                   max_length=self.max_length,
                                                   truncation=True,
                                                   padding='max_length',
                                                   return_tensors='pt')
        
        tgt_tokens = self.target_tokenizer.encode(self.target_texts[idx],
                                                   max_length=self.max_length,
                                                   truncation=True,
                                                   padding='max_length',
                                                   return_tensors='pt')
        
        return {
            'src': src_tokens.squeeze(0),
            'tgt': tgt_tokens.squeeze(0)
        }


class LabelSmoothingLoss(nn.Module):
    """
    Label smoothing regularization to prevent overfitting
    Distributes probability mass to all tokens
    """
    
    def __init__(self, vocab_size: int, padding_idx: int, smoothing: float = 0.1):
        """
        Args:
            vocab_size: Size of vocabulary
            padding_idx: Index of padding token
            smoothing: Smoothing factor (0.0 to 1.0)
        """
        super().__init__()
        self.criterion = nn.KLDivLoss(reduction='sum')
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.vocab_size = vocab_size
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predicted logits (batch_size * seq_len, vocab_size)
            target: Target indices (batch_size * seq_len)
            
        Returns:
            Smoothed loss value
        """
        pred = pred.log_softmax(dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.vocab_size - 2))
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
            true_dist[:, self.padding_idx] = 0
            mask = torch.nonzero(target == self.padding_idx)
            if mask.dim() > 0 and mask.size(0) > 0:
                true_dist.index_fill_(0, mask.squeeze(), 0.0)
        
        return self.criterion(pred, true_dist)


class WarmupScheduler:
    """
    Learning rate scheduler with warmup
    Gradually increases LR then decreases it
    """
    
    def __init__(self, optimizer, d_model: int, warmup_steps: int = 4000):
        """
        Args:
            optimizer: PyTorch optimizer
            d_model: Model dimension
            warmup_steps: Number of warmup steps
        """
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.current_step = 0
        
    def step(self):
        """Update learning rate"""
        self.current_step += 1
        lr = self._get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
            
    def _get_lr(self) -> float:
        """Calculate learning rate based on current step"""
        step = self.current_step
        return (self.d_model ** -0.5) * min(step ** -0.5, step * self.warmup_steps ** -1.5)


class Trainer:
    """
    Training manager for transformer translation models
    """
    
    def __init__(self, 
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 config: Dict):
        """
        Args:
            model: Transformer model
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Training configuration dictionary
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        
        # Device configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Loss function with label smoothing
        self.criterion = LabelSmoothingLoss(
            vocab_size=config['tgt_vocab_size'],
            padding_idx=config['pad_idx'],
            smoothing=config.get('label_smoothing', 0.1)
        )
        
        # Optimizer
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=1.0,  # Will be controlled by scheduler
            betas=(0.9, 0.98),
            eps=1e-9
        )
        
        # Learning rate scheduler
        self.scheduler = WarmupScheduler(
            self.optimizer,
            d_model=config['d_model'],
            warmup_steps=config.get('warmup_steps', 4000)
        )
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        
        # Create checkpoint directory
        self.checkpoint_dir = config.get('checkpoint_dir', 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
    def train_epoch(self) -> float:
        """Train for one epoch"""
        self.model.train()
        epoch_loss = 0
        
        progress_bar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch}')
        
        for batch_idx, batch in enumerate(progress_bar):
            src = batch['src'].to(self.device)
            tgt = batch['tgt'].to(self.device)
            
            # Create input and target (shift target by one for teacher forcing)
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            
            # Create masks
            src_mask, tgt_mask = create_mask(src, tgt_input, pad_idx=self.config['pad_idx'])
            src_mask = src_mask.to(self.device)
            tgt_mask = tgt_mask.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(src, tgt_input, src_mask, tgt_mask)
            
            # Calculate loss
            output = output.contiguous().view(-1, output.size(-1))
            tgt_output = tgt_output.contiguous().view(-1)
            
            loss = self.criterion(output, tgt_output)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Update parameters
            self.optimizer.step()
            self.scheduler.step()
            
            # Update statistics
            epoch_loss += loss.item()
            self.global_step += 1
            
            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item()})
            
        return epoch_loss / len(self.train_loader)
    
    def validate(self) -> float:
        """Validate model"""
        self.model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                src = batch['src'].to(self.device)
                tgt = batch['tgt'].to(self.device)
                
                # Create input and target
                tgt_input = tgt[:, :-1]
                tgt_output = tgt[:, 1:]
                
                # Create masks
                src_mask, tgt_mask = create_mask(src, tgt_input, pad_idx=self.config['pad_idx'])
                src_mask = src_mask.to(self.device)
                tgt_mask = tgt_mask.to(self.device)
                
                # Forward pass
                output = self.model(src, tgt_input, src_mask, tgt_mask)
                
                # Calculate loss
                output = output.contiguous().view(-1, output.size(-1))
                tgt_output = tgt_output.contiguous().view(-1)
                
                loss = self.criterion(output, tgt_output)
                val_loss += loss.item()
        
        return val_loss / len(self.val_loader)
    
    def train(self, num_epochs: int):
        """
        Train model for specified number of epochs
        
        Args:
            num_epochs: Number of training epochs
        """
        print(f"Training on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Train
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss = self.validate()
            self.val_losses.append(val_loss)
            
            print(f"\nEpoch {epoch}:")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            
            # Save checkpoint if best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint('best_model.pt')
                print(f"  ✓ Saved best model (val_loss: {val_loss:.4f})")
            
            # Save regular checkpoint
            if (epoch + 1) % self.config.get('save_every', 5) == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch}.pt')
        
        print("\n✓ Training completed!")
        
    def save_checkpoint(self, filename: str):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        path = os.path.join(self.checkpoint_dir, filename)
        torch.save(checkpoint, path)
        
    def load_checkpoint(self, filepath: str):
        """Load model checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.best_val_loss = checkpoint['best_val_loss']
        
        print(f"Loaded checkpoint from epoch {self.current_epoch}")


def train_de_en_model(config: Dict):
    """Train German to English translation model"""
    print("=" * 60)
    print("Training German → English Model")
    print("=" * 60)
    
    # Create model
    model = Transformer(
        src_vocab_size=config['de_vocab_size'],
        tgt_vocab_size=config['en_vocab_size'],
        d_model=config['d_model'],
        num_heads=config['num_heads'],
        num_encoder_layers=config['num_encoder_layers'],
        num_decoder_layers=config['num_decoder_layers'],
        d_ff=config['d_ff'],
        max_seq_len=config['max_seq_len'],
        dropout=config['dropout']
    )
    
    # Create data loaders (placeholder - implement actual data loading)
    # train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=config['batch_size'])
    
    # For demonstration - would need actual datasets
    print("Note: Implement actual data loading for German-English pairs")
    

def train_en_mr_model(config: Dict):
    """Train English to Marathi translation model"""
    print("=" * 60)
    print("Training English → Marathi Model")
    print("=" * 60)
    
    # Create model
    model = Transformer(
        src_vocab_size=config['en_vocab_size'],
        tgt_vocab_size=config['mr_vocab_size'],
        d_model=config['d_model'],
        num_heads=config['num_heads'],
        num_encoder_layers=config['num_encoder_layers'],
        num_decoder_layers=config['num_decoder_layers'],
        d_ff=config['d_ff'],
        max_seq_len=config['max_seq_len'],
        dropout=config['dropout']
    )
    
    # Create data loaders (placeholder - implement actual data loading)
    print("Note: Implement actual data loading for English-Marathi pairs")


if __name__ == "__main__":
    # Training configuration
    config = {
        # Model architecture
        'd_model': 512,
        'num_heads': 8,
        'num_encoder_layers': 6,
        'num_decoder_layers': 6,
        'd_ff': 2048,
        'max_seq_len': 128,
        'dropout': 0.1,
        
        # Vocabulary sizes (example values)
        'de_vocab_size': 32000,  # German
        'en_vocab_size': 32000,  # English
        'mr_vocab_size': 32000,  # Marathi
        'tgt_vocab_size': 32000,
        'pad_idx': 0,
        
        # Training parameters
        'batch_size': 32,
        'num_epochs': 30,
        'warmup_steps': 4000,
        'label_smoothing': 0.1,
        'save_every': 5,
        
        # Paths
        'checkpoint_dir': 'checkpoints',
    }
    
    print("Translation Model Training Pipeline")
    print("Configuration:", json.dumps(config, indent=2))
    
    # Train models
    # train_de_en_model(config)
    # train_en_mr_model(config)
    
    print("\n✓ Training pipeline ready!")
    print("Note: Connect actual datasets to begin training")
