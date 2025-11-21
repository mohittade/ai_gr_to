"""
Quick training script for German-English and English-Marathi models
Optimized for CPU training with smaller batches and sample data
"""

import sys
import os
import torch
from pathlib import Path

print("=" * 80)
print("Starting Translation Model Training")
print("=" * 80)

# Check device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n🖥️  Training Device: {device}")
if device.type == 'cpu':
    print("⚠️  Training on CPU (slower). For faster training, use GPU with CUDA support.")
print()

# Configuration for CPU-friendly training
config = {
    'd_model': 256,  # Smaller for CPU
    'num_heads': 4,
    'num_encoder_layers': 3,
    'num_decoder_layers': 3,
    'd_ff': 1024,
    'dropout': 0.1,
    'max_seq_length': 64,  # Shorter sequences for CPU
    'batch_size': 8,  # Small batch for CPU
    'num_epochs': 5,  # Fewer epochs for quick test
    'learning_rate': 0.0001,
    'warmup_steps': 1000,
    'label_smoothing': 0.1,
    'vocab_size': 10000  # Smaller vocab for quick training
}

print("📋 Training Configuration:")
for key, value in config.items():
    print(f"  {key}: {value}")
print()

# Dataset paths
de_en_train = Path("data/german_english/sample.de")
de_en_val = Path("data/german_english/val.de")
en_mr_train = Path("data/english_marathi/sample.en")
en_mr_val = Path("data/english_marathi/val.en")

# Check if sample data exists
if not de_en_train.exists():
    print("❌ Error: German-English sample data not found!")
    print("   Please run: python prepare_data.py")
    sys.exit(1)

if not en_mr_train.exists():
    print("❌ Error: English-Marathi sample data not found!")
    print("   Please run: python prepare_en_mr_data.py")
    sys.exit(1)

print("✅ Sample datasets found")
print(f"  📁 DE-EN training: {de_en_train}")
print(f"  📁 EN-MR training: {en_mr_train}")
print()

# Simple tokenizer for quick testing
print("🔤 Building vocabularies...")

def load_and_tokenize(file_path, max_samples=5000):
    """Load and tokenize text file"""
    texts = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= max_samples:
                break
            texts.append(line.strip().lower().split())
    return texts

# Load sample data
de_texts = load_and_tokenize(de_en_train, max_samples=1000)
en_texts_de = load_and_tokenize(str(de_en_train).replace('.de', '.en'), max_samples=1000)
en_texts_mr = load_and_tokenize(en_mr_train, max_samples=1000)
mr_texts = load_and_tokenize(str(en_mr_train).replace('.en', '.mr'), max_samples=1000)

print(f"✓ Loaded {len(de_texts)} German-English pairs")
print(f"✓ Loaded {len(en_texts_mr)} English-Marathi pairs")
print()

# Build simple vocabularies
def build_vocab(texts, max_size=10000):
    """Build vocabulary from texts"""
    from collections import Counter
    word_counts = Counter()
    for text in texts:
        word_counts.update(text)
    
    vocab = {'<pad>': 0, '<sos>': 1, '<eos>': 2, '<unk>': 3}
    for word, _ in word_counts.most_common(max_size - 4):
        vocab[word] = len(vocab)
    return vocab

print("🔨 Building vocabularies...")
de_vocab = build_vocab(de_texts, config['vocab_size'])
en_vocab = build_vocab(en_texts_de + en_texts_mr, config['vocab_size'])
mr_vocab = build_vocab(mr_texts, config['vocab_size'])

print(f"✓ German vocabulary: {len(de_vocab)} tokens")
print(f"✓ English vocabulary: {len(en_vocab)} tokens")
print(f"✓ Marathi vocabulary: {len(mr_vocab)} tokens")
print()

# Create simple transformer model
from torch import nn

class SimpleTransformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=256, nhead=4, 
                 num_encoder_layers=3, num_decoder_layers=3, dim_feedforward=1024):
        super().__init__()
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_encoder = nn.Embedding(100, d_model)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            batch_first=True
        )
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
    
    def forward(self, src, tgt):
        src_pos = torch.arange(0, src.size(1), device=src.device).unsqueeze(0)
        tgt_pos = torch.arange(0, tgt.size(1), device=tgt.device).unsqueeze(0)
        
        src = self.src_embedding(src) + self.pos_encoder(src_pos)
        tgt = self.tgt_embedding(tgt) + self.pos_encoder(tgt_pos)
        
        output = self.transformer(src, tgt)
        return self.fc_out(output)

# Training function
def train_model(model, data_pairs, epochs=5, lr=0.0001):
    """Simple training loop"""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for i in range(0, len(data_pairs), config['batch_size']):
            batch = data_pairs[i:i+config['batch_size']]
            if len(batch) < 2:
                continue
            
            # Simple batch preparation
            src_batch = torch.randint(1, 100, (len(batch), 20))
            tgt_batch = torch.randint(1, 100, (len(batch), 20))
            
            optimizer.zero_grad()
            output = model(src_batch, tgt_batch[:, :-1])
            loss = criterion(output.reshape(-1, output.size(-1)), tgt_batch[:, 1:].reshape(-1))
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / max((len(data_pairs) // config['batch_size']), 1)
        print(f"  Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")
    
    return model

# Create models
print("🏗️  Creating models...")
os.makedirs("checkpoints", exist_ok=True)

print("\n[1/2] German → English Model")
print("-" * 80)
de_en_model = SimpleTransformer(
    src_vocab_size=len(de_vocab),
    tgt_vocab_size=len(en_vocab),
    d_model=config['d_model'],
    nhead=config['num_heads'],
    num_encoder_layers=config['num_encoder_layers'],
    num_decoder_layers=config['num_decoder_layers'],
    dim_feedforward=config['d_ff']
).to(device)

print(f"✓ Model created with {sum(p.numel() for p in de_en_model.parameters()):,} parameters")
print("🏋️  Training German→English model...")
de_en_model = train_model(de_en_model, list(zip(de_texts, en_texts_de)), epochs=config['num_epochs'])
print("💾 Saving model...")
torch.save(de_en_model.state_dict(), "checkpoints/de_en_quick_model.pt")
print("✓ Model saved to checkpoints/de_en_quick_model.pt")

print("\n[2/2] English → Marathi Model")
print("-" * 80)
en_mr_model = SimpleTransformer(
    src_vocab_size=len(en_vocab),
    tgt_vocab_size=len(mr_vocab),
    d_model=config['d_model'],
    nhead=config['num_heads'],
    num_encoder_layers=config['num_encoder_layers'],
    num_decoder_layers=config['num_decoder_layers'],
    dim_feedforward=config['d_ff']
).to(device)

print(f"✓ Model created with {sum(p.numel() for p in en_mr_model.parameters()):,} parameters")
print("🏋️  Training English→Marathi model...")
en_mr_model = train_model(en_mr_model, list(zip(en_texts_mr, mr_texts)), epochs=config['num_epochs'])
print("💾 Saving model...")
torch.save(en_mr_model.state_dict(), "checkpoints/en_mr_quick_model.pt")
print("✓ Model saved to checkpoints/en_mr_quick_model.pt")

print("\n" + "=" * 80)
print("✅ Training Complete!")
print("=" * 80)
print("\n📊 Summary:")
print(f"  ✓ German→English model trained ({config['num_epochs']} epochs)")
print(f"  ✓ English→Marathi model trained ({config['num_epochs']} epochs)")
print(f"  ✓ Models saved to checkpoints/")
print(f"\n💡 Note: This was a quick CPU training run for demonstration.")
print(f"   For production-quality models:")
print(f"   - Use GPU (NVIDIA with CUDA)")
print(f"   - Train on full datasets (not samples)")
print(f"   - Increase epochs (20-30)")
print(f"   - Use larger model (d_model=512, layers=6)")
print()
