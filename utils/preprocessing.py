"""
Data Preprocessing Module for German-English-Marathi Translation
Implements text cleaning, normalization, tokenization (BPE), and alignment
"""

import re
import unicodedata
from typing import List, Tuple, Dict
import sentencepiece as spm
from transformers import MarianTokenizer, AutoTokenizer


class DataPreprocessor:
    """
    Handles all preprocessing tasks for multilingual translation:
    - Text cleaning
    - Normalization
    - Tokenization (BPE)
    - Sentence alignment
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize the preprocessor with configuration
        
        Args:
            config: Dictionary containing preprocessing parameters
        """
        self.config = config or {}
        self.max_length = self.config.get('max_length', 128)
        self.min_length = self.config.get('min_length', 3)
        
    def clean_text(self, text: str) -> str:
        """
        Clean unwanted symbols, special characters, and non-language tokens
        
        Args:
            text: Raw input text
            
        Returns:
            Cleaned text string
        """
        # Remove URLs
        text = re.sub(r'http\S+|www\S+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove special characters but keep sentence structure
        text = re.sub(r'[^\w\s\.\,\?\!\;\:\-\'\"\u0900-\u097F]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def normalize_text(self, text: str, lowercase: bool = True) -> str:
        """
        Normalize text for consistency
        - Convert to lowercase (optional)
        - Handle Unicode inconsistencies
        - Normalize punctuation
        
        Args:
            text: Input text
            lowercase: Whether to convert to lowercase
            
        Returns:
            Normalized text
        """
        # Unicode normalization (NFKC form for compatibility)
        text = unicodedata.normalize('NFKC', text)
        
        # Optional lowercasing (may not be suitable for all languages)
        if lowercase:
            text = text.lower()
        
        # Normalize common punctuation
        text = text.replace("'", "'").replace('"', '"').replace('"', '"')
        text = text.replace('–', '-').replace('—', '-')
        
        # Remove zero-width characters
        text = text.replace('\u200b', '').replace('\u200c', '').replace('\u200d', '')
        
        return text.strip()
    
    def tokenize_bpe(self, text: str, model_name: str = "Helsinki-NLP/opus-mt-de-en") -> List[str]:
        """
        Tokenize text using Byte Pair Encoding (BPE)
        Handles rare words and reduces vocabulary size
        
        Args:
            text: Input text
            model_name: Pretrained tokenizer model name
            
        Returns:
            List of subword tokens
        """
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            tokens = tokenizer.tokenize(text)
            return tokens
        except Exception as e:
            print(f"Tokenization error: {e}")
            # Fallback to simple whitespace tokenization
            return text.split()
    
    def validate_pair(self, source: str, target: str) -> bool:
        """
        Validate that source and target sentences are valid parallel pairs
        
        Args:
            source: Source language sentence
            target: Target language sentence
            
        Returns:
            True if valid pair, False otherwise
        """
        # Check length constraints
        if len(source.split()) < self.min_length or len(target.split()) < self.min_length:
            return False
        
        if len(source.split()) > self.max_length or len(target.split()) > self.max_length:
            return False
        
        # Check length ratio (target shouldn't be 3x longer than source)
        length_ratio = len(target.split()) / max(len(source.split()), 1)
        if length_ratio > 3 or length_ratio < 0.3:
            return False
        
        return True
    
    def preprocess_dataset(self, 
                          source_texts: List[str], 
                          target_texts: List[str],
                          clean: bool = True,
                          normalize: bool = True) -> Tuple[List[str], List[str]]:
        """
        Full preprocessing pipeline for parallel corpus
        
        Args:
            source_texts: List of source language sentences
            target_texts: List of target language sentences
            clean: Whether to apply cleaning
            normalize: Whether to apply normalization
            
        Returns:
            Tuple of (processed_source, processed_target) lists
        """
        processed_source = []
        processed_target = []
        
        for src, tgt in zip(source_texts, target_texts):
            # Apply cleaning
            if clean:
                src = self.clean_text(src)
                tgt = self.clean_text(tgt)
            
            # Apply normalization
            if normalize:
                src = self.normalize_text(src)
                tgt = self.normalize_text(tgt)
            
            # Validate pair
            if self.validate_pair(src, tgt):
                processed_source.append(src)
                processed_target.append(tgt)
        
        print(f"Preprocessing complete: {len(processed_source)} valid pairs from {len(source_texts)} original pairs")
        return processed_source, processed_target


class BilingualDataset:
    """
    Dataset class for managing bilingual parallel corpora
    """
    
    def __init__(self, source_file: str, target_file: str, preprocessor: DataPreprocessor):
        """
        Initialize bilingual dataset
        
        Args:
            source_file: Path to source language file
            target_file: Path to target language file
            preprocessor: DataPreprocessor instance
        """
        self.preprocessor = preprocessor
        self.source_texts = self._load_file(source_file)
        self.target_texts = self._load_file(target_file)
        
        # Preprocess data
        self.source_texts, self.target_texts = preprocessor.preprocess_dataset(
            self.source_texts, self.target_texts
        )
    
    def _load_file(self, filepath: str) -> List[str]:
        """Load text file line by line"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return [line.strip() for line in f if line.strip()]
        except FileNotFoundError:
            print(f"Warning: File not found - {filepath}")
            return []
    
    def __len__(self) -> int:
        return len(self.source_texts)
    
    def __getitem__(self, idx: int) -> Tuple[str, str]:
        return self.source_texts[idx], self.target_texts[idx]
    
    def get_batch(self, start_idx: int, batch_size: int) -> Tuple[List[str], List[str]]:
        """Get a batch of parallel sentences"""
        end_idx = min(start_idx + batch_size, len(self))
        return (
            self.source_texts[start_idx:end_idx],
            self.target_texts[start_idx:end_idx]
        )


if __name__ == "__main__":
    # Example usage
    preprocessor = DataPreprocessor(config={'max_length': 150, 'min_length': 3})
    
    # Test text cleaning
    sample_text = "Visit http://example.com for more info! Email: test@example.com @@#$%"
    print("Original:", sample_text)
    print("Cleaned:", preprocessor.clean_text(sample_text))
    
    # Test normalization
    sample_text = "  The café's 'special' deal—50% off!  "
    print("\nOriginal:", sample_text)
    print("Normalized:", preprocessor.normalize_text(sample_text))
    
    # Test validation
    src = "Das ist ein Beispielsatz"
    tgt = "This is an example sentence"
    print(f"\nValid pair: {preprocessor.validate_pair(src, tgt)}")
