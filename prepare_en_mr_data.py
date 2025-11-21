"""
Data preparation script for English-Marathi corpus
Splits data into train/validation/test sets and performs basic quality filtering
"""

import os
import random
from pathlib import Path
from tqdm import tqdm
import json

def load_parallel_data(en_file, mr_file):
    """
    Load parallel sentences from English and Marathi files
    """
    print(f"Loading data from:\n  {en_file}\n  {mr_file}")
    
    with open(en_file, 'r', encoding='utf-8') as f_en:
        en_lines = f_en.readlines()
    
    with open(mr_file, 'r', encoding='utf-8') as f_mr:
        mr_lines = f_mr.readlines()
    
    print(f"✓ Loaded {len(en_lines):,} English sentences")
    print(f"✓ Loaded {len(mr_lines):,} Marathi sentences")
    
    if len(en_lines) != len(mr_lines):
        print(f"⚠ Warning: Mismatched line counts! Using minimum: {min(len(en_lines), len(mr_lines))}")
        min_len = min(len(en_lines), len(mr_lines))
        en_lines = en_lines[:min_len]
        mr_lines = mr_lines[:min_len]
    
    return en_lines, mr_lines

def filter_pairs(en_lines, mr_lines, min_length=3, max_length=128, max_length_ratio=3.0):
    """
    Filter sentence pairs based on quality criteria
    Note: Marathi uses Devanagari script which has different tokenization characteristics
    """
    print("\nFiltering sentence pairs...")
    filtered_en = []
    filtered_mr = []
    
    stats = {
        'total': len(en_lines),
        'too_short': 0,
        'too_long': 0,
        'length_ratio': 0,
        'empty': 0,
        'kept': 0
    }
    
    for en, mr in tqdm(zip(en_lines, mr_lines), total=len(en_lines), desc="Filtering"):
        en = en.strip()
        mr = mr.strip()
        
        # Skip empty lines
        if not en or not mr:
            stats['empty'] += 1
            continue
        
        # Token count (rough estimate using spaces)
        en_tokens = len(en.split())
        mr_tokens = len(mr.split())
        
        # Skip if too short
        if en_tokens < min_length or mr_tokens < min_length:
            stats['too_short'] += 1
            continue
        
        # Skip if too long
        if en_tokens > max_length or mr_tokens > max_length:
            stats['too_long'] += 1
            continue
        
        # Skip if length ratio is too different
        # Note: English-Marathi can have larger length differences due to script differences
        length_ratio = max(en_tokens, mr_tokens) / max(min(en_tokens, mr_tokens), 1)
        if length_ratio > max_length_ratio:
            stats['length_ratio'] += 1
            continue
        
        filtered_en.append(en)
        filtered_mr.append(mr)
        stats['kept'] += 1
    
    print(f"\n📊 Filtering Statistics:")
    print(f"  Total pairs: {stats['total']:,}")
    print(f"  Kept: {stats['kept']:,} ({stats['kept']/stats['total']*100:.1f}%)")
    print(f"  Removed:")
    print(f"    - Too short: {stats['too_short']:,}")
    print(f"    - Too long: {stats['too_long']:,}")
    print(f"    - Length ratio: {stats['length_ratio']:,}")
    print(f"    - Empty: {stats['empty']:,}")
    
    return filtered_en, filtered_mr

def split_data(en_lines, mr_lines, train_ratio=0.90, val_ratio=0.05, test_ratio=0.05, seed=42):
    """
    Split data into train/validation/test sets
    """
    print(f"\nSplitting data (train={train_ratio}, val={val_ratio}, test={test_ratio})...")
    
    # Set seed for reproducibility
    random.seed(seed)
    
    # Create paired list and shuffle
    pairs = list(zip(en_lines, mr_lines))
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
    en_path = output_dir / f"{split_name}.en"
    mr_path = output_dir / f"{split_name}.mr"
    
    print(f"\nSaving {split_name} split...")
    
    with open(en_path, 'w', encoding='utf-8') as f_en, \
         open(mr_path, 'w', encoding='utf-8') as f_mr:
        for en, mr in pairs:
            f_en.write(en + '\n')
            f_mr.write(mr + '\n')
    
    print(f"✓ Saved to:\n  {en_path}\n  {mr_path}")

def create_sample_data(train_pairs, output_dir, sample_size=5000):
    """
    Create a small sample dataset for quick testing
    """
    print(f"\nCreating sample dataset ({sample_size} pairs)...")
    
    sample_pairs = random.sample(train_pairs, min(sample_size, len(train_pairs)))
    
    en_path = output_dir / "sample.en"
    mr_path = output_dir / "sample.mr"
    
    with open(en_path, 'w', encoding='utf-8') as f_en, \
         open(mr_path, 'w', encoding='utf-8') as f_mr:
        for en, mr in sample_pairs:
            f_en.write(en + '\n')
            f_mr.write(mr + '\n')
    
    print(f"✓ Sample saved to:\n  {en_path}\n  {mr_path}")

def show_examples(pairs, num_examples=5):
    """
    Display sample translations
    """
    print(f"\n📝 Sample Translations:")
    print("-" * 80)
    
    for i, (en, mr) in enumerate(pairs[:num_examples], 1):
        print(f"\nExample {i}:")
        print(f"  🇬🇧 EN: {en}")
        print(f"  🇮🇳 MR: {mr}")
    
    print("-" * 80)

def save_stats(stats, output_file):
    """
    Save dataset statistics to JSON
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)
    print(f"✓ Statistics saved to: {output_file}")

def main():
    print("=" * 80)
    print("English-Marathi Data Preparation")
    print("=" * 80)
    
    # Paths
    data_dir = Path("data/english_marathi")
    en_file = data_dir / "train.en"
    mr_file = data_dir / "train.mr"
    
    # Check if files exist
    if not en_file.exists() or not mr_file.exists():
        print(f"❌ Error: Dataset files not found!")
        print(f"   Expected: {en_file} and {mr_file}")
        return
    
    # Load data
    en_lines, mr_lines = load_parallel_data(en_file, mr_file)
    
    # Filter pairs (more lenient for Marathi due to script differences)
    en_filtered, mr_filtered = filter_pairs(
        en_lines, 
        mr_lines,
        min_length=3,
        max_length=128,
        max_length_ratio=3.0  # More lenient for English-Marathi
    )
    
    # Split data
    train_pairs, val_pairs, test_pairs = split_data(
        en_filtered,
        mr_filtered,
        train_ratio=0.90,
        val_ratio=0.05,
        test_ratio=0.05
    )
    
    # Show examples
    show_examples(train_pairs, num_examples=5)
    
    # Save splits
    save_split(train_pairs, data_dir, "train_split")
    save_split(val_pairs, data_dir, "val")
    save_split(test_pairs, data_dir, "test")
    
    # Create sample dataset
    create_sample_data(train_pairs, data_dir, sample_size=5000)
    
    # Save statistics
    stats = {
        'original_size': len(en_lines),
        'filtered_size': len(en_filtered),
        'train_size': len(train_pairs),
        'val_size': len(val_pairs),
        'test_size': len(test_pairs),
        'sample_size': min(5000, len(train_pairs)),
        'filter_rate': len(en_filtered) / len(en_lines),
        'splits': {
            'train': 0.90,
            'val': 0.05,
            'test': 0.05
        },
        'note': 'English-Marathi corpus from government press releases'
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
    print(f"  {data_dir}/train_split.en & train_split.mr")
    print(f"  {data_dir}/val.en & val.mr")
    print(f"  {data_dir}/test.en & test.mr")
    print(f"  {data_dir}/sample.en & sample.mr")
    print(f"  {data_dir}/dataset_stats.json")
    
    print(f"\n🚀 Next steps:")
    print(f"  1. Train German→English model: python examples/train_example.py --model de-en")
    print(f"  2. Train English→Marathi model: python examples/train_example.py --model en-mr")
    print(f"  3. Test full pipeline: python examples/basic_translation.py")
    print("")

if __name__ == "__main__":
    main()
