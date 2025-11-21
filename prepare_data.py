"""
Data preparation script for German-English Europarl corpus
Splits data into train/validation/test sets and performs basic quality filtering
"""

import os
import random
from pathlib import Path
from tqdm import tqdm
import json

def load_parallel_data(de_file, en_file):
    """
    Load parallel sentences from German and English files
    """
    print(f"Loading data from:\n  {de_file}\n  {en_file}")
    
    with open(de_file, 'r', encoding='utf-8') as f_de:
        de_lines = f_de.readlines()
    
    with open(en_file, 'r', encoding='utf-8') as f_en:
        en_lines = f_en.readlines()
    
    print(f"✓ Loaded {len(de_lines):,} German sentences")
    print(f"✓ Loaded {len(en_lines):,} English sentences")
    
    if len(de_lines) != len(en_lines):
        print(f"⚠ Warning: Mismatched line counts! Using minimum: {min(len(de_lines), len(en_lines))}")
        min_len = min(len(de_lines), len(en_lines))
        de_lines = de_lines[:min_len]
        en_lines = en_lines[:min_len]
    
    return de_lines, en_lines

def filter_pairs(de_lines, en_lines, min_length=3, max_length=128, max_length_ratio=2.0):
    """
    Filter sentence pairs based on quality criteria
    """
    print("\nFiltering sentence pairs...")
    filtered_de = []
    filtered_en = []
    
    stats = {
        'total': len(de_lines),
        'too_short': 0,
        'too_long': 0,
        'length_ratio': 0,
        'empty': 0,
        'kept': 0
    }
    
    for de, en in tqdm(zip(de_lines, en_lines), total=len(de_lines), desc="Filtering"):
        de = de.strip()
        en = en.strip()
        
        # Skip empty lines
        if not de or not en:
            stats['empty'] += 1
            continue
        
        # Token count (rough estimate using spaces)
        de_tokens = len(de.split())
        en_tokens = len(en.split())
        
        # Skip if too short
        if de_tokens < min_length or en_tokens < min_length:
            stats['too_short'] += 1
            continue
        
        # Skip if too long
        if de_tokens > max_length or en_tokens > max_length:
            stats['too_long'] += 1
            continue
        
        # Skip if length ratio is too different
        length_ratio = max(de_tokens, en_tokens) / max(min(de_tokens, en_tokens), 1)
        if length_ratio > max_length_ratio:
            stats['length_ratio'] += 1
            continue
        
        filtered_de.append(de)
        filtered_en.append(en)
        stats['kept'] += 1
    
    print(f"\n📊 Filtering Statistics:")
    print(f"  Total pairs: {stats['total']:,}")
    print(f"  Kept: {stats['kept']:,} ({stats['kept']/stats['total']*100:.1f}%)")
    print(f"  Removed:")
    print(f"    - Too short: {stats['too_short']:,}")
    print(f"    - Too long: {stats['too_long']:,}")
    print(f"    - Length ratio: {stats['length_ratio']:,}")
    print(f"    - Empty: {stats['empty']:,}")
    
    return filtered_de, filtered_en

def split_data(de_lines, en_lines, train_ratio=0.95, val_ratio=0.025, test_ratio=0.025, seed=42):
    """
    Split data into train/validation/test sets
    """
    print(f"\nSplitting data (train={train_ratio}, val={val_ratio}, test={test_ratio})...")
    
    # Set seed for reproducibility
    random.seed(seed)
    
    # Create paired list and shuffle
    pairs = list(zip(de_lines, en_lines))
    random.shuffle(pairs)
    
    # Calculate split indices
    n = len(pairs)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    
    # Split
    train_pairs = pairs[:train_end]
    val_pairs = pairs[train_end:val_end]
    test_pairs = pairs[val_end:]
    
    print(f"✓ Train: {len(train_pairs):,} pairs")
    print(f"✓ Validation: {len(val_pairs):,} pairs")
    print(f"✓ Test: {len(test_pairs):,} pairs")
    
    return train_pairs, val_pairs, test_pairs

def save_split(pairs, output_dir, split_name):
    """
    Save a data split to files
    """
    de_path = output_dir / f"{split_name}.de"
    en_path = output_dir / f"{split_name}.en"
    
    print(f"\nSaving {split_name} split...")
    
    with open(de_path, 'w', encoding='utf-8') as f_de, \
         open(en_path, 'w', encoding='utf-8') as f_en:
        for de, en in pairs:
            f_de.write(de + '\n')
            f_en.write(en + '\n')
    
    print(f"✓ Saved to:\n  {de_path}\n  {en_path}")

def create_sample_data(train_pairs, output_dir, sample_size=10000):
    """
    Create a small sample dataset for quick testing
    """
    print(f"\nCreating sample dataset ({sample_size} pairs)...")
    
    sample_pairs = random.sample(train_pairs, min(sample_size, len(train_pairs)))
    
    de_path = output_dir / "sample.de"
    en_path = output_dir / "sample.en"
    
    with open(de_path, 'w', encoding='utf-8') as f_de, \
         open(en_path, 'w', encoding='utf-8') as f_en:
        for de, en in sample_pairs:
            f_de.write(de + '\n')
            f_en.write(en + '\n')
    
    print(f"✓ Sample saved to:\n  {de_path}\n  {en_path}")

def save_stats(stats, output_file):
    """
    Save dataset statistics to JSON
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)
    print(f"✓ Statistics saved to: {output_file}")

def main():
    print("=" * 80)
    print("German-English Data Preparation")
    print("=" * 80)
    
    # Paths
    data_dir = Path("data/german_english")
    de_file = data_dir / "europarl.de"
    en_file = data_dir / "europarl.en"
    
    # Check if files exist
    if not de_file.exists() or not en_file.exists():
        print(f"❌ Error: Dataset files not found!")
        print(f"   Expected: {de_file} and {en_file}")
        return
    
    # Load data
    de_lines, en_lines = load_parallel_data(de_file, en_file)
    
    # Filter pairs
    de_filtered, en_filtered = filter_pairs(
        de_lines, 
        en_lines,
        min_length=3,
        max_length=128,
        max_length_ratio=2.0
    )
    
    # Split data
    train_pairs, val_pairs, test_pairs = split_data(
        de_filtered,
        en_filtered,
        train_ratio=0.95,
        val_ratio=0.025,
        test_ratio=0.025
    )
    
    # Save splits
    save_split(train_pairs, data_dir, "train")
    save_split(val_pairs, data_dir, "val")
    save_split(test_pairs, data_dir, "test")
    
    # Create sample dataset
    create_sample_data(train_pairs, data_dir, sample_size=10000)
    
    # Save statistics
    stats = {
        'original_size': len(de_lines),
        'filtered_size': len(de_filtered),
        'train_size': len(train_pairs),
        'val_size': len(val_pairs),
        'test_size': len(test_pairs),
        'sample_size': min(10000, len(train_pairs)),
        'filter_rate': len(de_filtered) / len(de_lines),
        'splits': {
            'train': 0.95,
            'val': 0.025,
            'test': 0.025
        }
    }
    
    save_stats(stats, data_dir / "dataset_stats.json")
    
    # Print summary
    print("\n" + "=" * 80)
    print("✅ Data preparation complete!")
    print("=" * 80)
    print(f"\n📊 Summary:")
    print(f"  Original: {stats['original_size']:,} pairs")
    print(f"  After filtering: {stats['filtered_size']:,} pairs ({stats['filter_rate']*100:.1f}%)")
    print(f"  Train: {stats['train_size']:,} pairs")
    print(f"  Validation: {stats['val_size']:,} pairs")
    print(f"  Test: {stats['test_size']:,} pairs")
    print(f"  Sample: {stats['sample_size']:,} pairs (for quick testing)")
    
    print(f"\n📁 Output files:")
    print(f"  {data_dir}/train.de & train.en")
    print(f"  {data_dir}/val.de & val.en")
    print(f"  {data_dir}/test.de & test.en")
    print(f"  {data_dir}/sample.de & sample.en")
    print(f"  {data_dir}/dataset_stats.json")
    
    print(f"\n🚀 Next step:")
    print(f"  Train model: python examples/train_example.py --model de-en")
    print("")

if __name__ == "__main__":
    main()
