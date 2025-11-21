"""
Evaluation Metrics for Neural Machine Translation
Implements BLEU, METEOR, and utilities for human evaluation
"""

import numpy as np
from typing import List, Dict, Tuple
from collections import Counter
import math


class BLEUScore:
    """
    BLEU (Bilingual Evaluation Understudy) Score
    Measures n-gram overlap between hypothesis and reference translations
    """
    
    def __init__(self, max_n: int = 4, weights: List[float] = None):
        """
        Args:
            max_n: Maximum n-gram order (default: 4 for BLEU-4)
            weights: Weights for different n-grams (default: uniform)
        """
        self.max_n = max_n
        self.weights = weights or [1.0 / max_n] * max_n
        
    def _get_ngrams(self, tokens: List[str], n: int) -> Counter:
        """Extract n-grams from token list"""
        ngrams = []
        for i in range(len(tokens) - n + 1):
            ngrams.append(tuple(tokens[i:i+n]))
        return Counter(ngrams)
    
    def _modified_precision(self, hypothesis: List[str], reference: List[str], n: int) -> float:
        """
        Calculate modified n-gram precision
        Clips counts to avoid rewarding repetition
        """
        hyp_ngrams = self._get_ngrams(hypothesis, n)
        ref_ngrams = self._get_ngrams(reference, n)
        
        if sum(hyp_ngrams.values()) == 0:
            return 0.0
        
        # Clip counts
        clipped_counts = 0
        for ngram, count in hyp_ngrams.items():
            clipped_counts += min(count, ref_ngrams.get(ngram, 0))
        
        precision = clipped_counts / sum(hyp_ngrams.values())
        return precision
    
    def _brevity_penalty(self, hyp_len: int, ref_len: int) -> float:
        """
        Brevity penalty to penalize short translations
        """
        if hyp_len >= ref_len:
            return 1.0
        return math.exp(1 - (ref_len / hyp_len))
    
    def compute(self, hypothesis: str, reference: str) -> float:
        """
        Compute BLEU score for single hypothesis-reference pair
        
        Args:
            hypothesis: Translated sentence
            reference: Reference translation
            
        Returns:
            BLEU score (0 to 1)
        """
        hyp_tokens = hypothesis.strip().split()
        ref_tokens = reference.strip().split()
        
        if len(hyp_tokens) == 0:
            return 0.0
        
        # Calculate precisions for all n-grams
        precisions = []
        for n in range(1, self.max_n + 1):
            precision = self._modified_precision(hyp_tokens, ref_tokens, n)
            if precision == 0:
                # Smoothing for zero counts
                precision = 1e-10
            precisions.append(precision)
        
        # Geometric mean of precisions
        log_precisions = [w * math.log(p) for w, p in zip(self.weights, precisions)]
        geo_mean = math.exp(sum(log_precisions))
        
        # Apply brevity penalty
        bp = self._brevity_penalty(len(hyp_tokens), len(ref_tokens))
        
        bleu = bp * geo_mean
        return bleu
    
    def corpus_bleu(self, hypotheses: List[str], references: List[str]) -> float:
        """
        Compute corpus-level BLEU score
        
        Args:
            hypotheses: List of translated sentences
            references: List of reference translations
            
        Returns:
            Corpus BLEU score (0 to 1)
        """
        total_hyp_len = 0
        total_ref_len = 0
        clipped_counts = [0] * self.max_n
        total_counts = [0] * self.max_n
        
        for hyp, ref in zip(hypotheses, references):
            hyp_tokens = hyp.strip().split()
            ref_tokens = ref.strip().split()
            
            total_hyp_len += len(hyp_tokens)
            total_ref_len += len(ref_tokens)
            
            # Count n-grams
            for n in range(1, self.max_n + 1):
                hyp_ngrams = self._get_ngrams(hyp_tokens, n)
                ref_ngrams = self._get_ngrams(ref_tokens, n)
                
                for ngram, count in hyp_ngrams.items():
                    clipped_counts[n-1] += min(count, ref_ngrams.get(ngram, 0))
                    total_counts[n-1] += count
        
        # Calculate precisions
        precisions = []
        for n in range(self.max_n):
            if total_counts[n] == 0:
                precision = 0
            else:
                precision = clipped_counts[n] / total_counts[n]
            
            if precision == 0:
                precision = 1e-10
            precisions.append(precision)
        
        # Geometric mean
        log_precisions = [w * math.log(p) for w, p in zip(self.weights, precisions)]
        geo_mean = math.exp(sum(log_precisions))
        
        # Brevity penalty
        bp = self._brevity_penalty(total_hyp_len, total_ref_len)
        
        corpus_bleu = bp * geo_mean
        return corpus_bleu


class METEORScore:
    """
    METEOR (Metric for Evaluation of Translation with Explicit ORdering)
    Uses stemming and synonyms for better semantic evaluation
    Note: Simplified version without WordNet synonyms
    """
    
    def __init__(self, alpha: float = 0.9, beta: float = 3.0, gamma: float = 0.5):
        """
        Args:
            alpha: Weight for precision vs recall
            beta: Weight for fragmentation penalty
            gamma: Fragmentation parameter
        """
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
    
    def _simple_stem(self, word: str) -> str:
        """Simple stemming (remove common suffixes)"""
        suffixes = ['ing', 'ed', 'es', 's', 'ly']
        for suffix in suffixes:
            if word.endswith(suffix) and len(word) > len(suffix) + 2:
                return word[:-len(suffix)]
        return word
    
    def _align_words(self, hypothesis: List[str], reference: List[str]) -> int:
        """
        Count aligned words between hypothesis and reference
        Uses exact match and stemming
        """
        hyp_stems = [self._simple_stem(w.lower()) for w in hypothesis]
        ref_stems = [self._simple_stem(w.lower()) for w in reference]
        
        matched = 0
        used_ref = set()
        
        # Exact matches first
        for i, hyp_word in enumerate(hypothesis):
            for j, ref_word in enumerate(reference):
                if j not in used_ref and hyp_word.lower() == ref_word.lower():
                    matched += 1
                    used_ref.add(j)
                    break
        
        # Stem matches
        used_ref_stems = set()
        for i, hyp_stem in enumerate(hyp_stems):
            for j, ref_stem in enumerate(ref_stems):
                if j not in used_ref and j not in used_ref_stems and hyp_stem == ref_stem:
                    matched += 1
                    used_ref_stems.add(j)
                    break
        
        return matched
    
    def compute(self, hypothesis: str, reference: str) -> float:
        """
        Compute METEOR score for single hypothesis-reference pair
        
        Args:
            hypothesis: Translated sentence
            reference: Reference translation
            
        Returns:
            METEOR score (0 to 1)
        """
        hyp_tokens = hypothesis.strip().split()
        ref_tokens = reference.strip().split()
        
        if len(hyp_tokens) == 0 or len(ref_tokens) == 0:
            return 0.0
        
        # Count aligned words
        matches = self._align_words(hyp_tokens, ref_tokens)
        
        # Precision and recall
        precision = matches / len(hyp_tokens) if len(hyp_tokens) > 0 else 0
        recall = matches / len(ref_tokens) if len(ref_tokens) > 0 else 0
        
        if precision == 0 and recall == 0:
            return 0.0
        
        # F-mean
        f_mean = (precision * recall) / (self.alpha * precision + (1 - self.alpha) * recall)
        
        # Simplified fragmentation penalty (set to 1 for now)
        penalty = 1.0
        
        meteor = f_mean * penalty
        return meteor


class TranslationEvaluator:
    """
    Comprehensive evaluation for translation quality
    """
    
    def __init__(self):
        """Initialize evaluator with BLEU and METEOR metrics"""
        self.bleu_scorer = BLEUScore(max_n=4)
        self.meteor_scorer = METEORScore()
    
    def evaluate(self, hypotheses: List[str], references: List[str]) -> Dict[str, float]:
        """
        Evaluate translation quality with multiple metrics
        
        Args:
            hypotheses: List of translated sentences
            references: List of reference translations
            
        Returns:
            Dictionary containing various metrics
        """
        if len(hypotheses) != len(references):
            raise ValueError("Number of hypotheses and references must match")
        
        # BLEU score
        bleu_score = self.bleu_scorer.corpus_bleu(hypotheses, references)
        
        # Individual BLEU scores
        individual_bleu = [
            self.bleu_scorer.compute(hyp, ref)
            for hyp, ref in zip(hypotheses, references)
        ]
        
        # METEOR scores
        individual_meteor = [
            self.meteor_scorer.compute(hyp, ref)
            for hyp, ref in zip(hypotheses, references)
        ]
        
        results = {
            'corpus_bleu': bleu_score,
            'avg_bleu': np.mean(individual_bleu),
            'avg_meteor': np.mean(individual_meteor),
            'total_sentences': len(hypotheses)
        }
        
        return results
    
    def generate_evaluation_report(self, 
                                   hypotheses: List[str], 
                                   references: List[str],
                                   source_texts: List[str] = None) -> str:
        """
        Generate detailed evaluation report
        
        Args:
            hypotheses: Translated sentences
            references: Reference translations
            source_texts: Optional source sentences
            
        Returns:
            Formatted report string
        """
        results = self.evaluate(hypotheses, references)
        
        report = "=" * 60 + "\n"
        report += "TRANSLATION EVALUATION REPORT\n"
        report += "=" * 60 + "\n\n"
        
        report += f"Total Sentences: {results['total_sentences']}\n\n"
        report += f"BLEU Score (Corpus): {results['corpus_bleu']:.4f}\n"
        report += f"BLEU Score (Average): {results['avg_bleu']:.4f}\n"
        report += f"METEOR Score (Average): {results['avg_meteor']:.4f}\n\n"
        
        # Sample translations
        report += "=" * 60 + "\n"
        report += "SAMPLE TRANSLATIONS (First 5)\n"
        report += "=" * 60 + "\n\n"
        
        for i in range(min(5, len(hypotheses))):
            if source_texts:
                report += f"Source:     {source_texts[i]}\n"
            report += f"Hypothesis: {hypotheses[i]}\n"
            report += f"Reference:  {references[i]}\n"
            report += f"BLEU:       {self.bleu_scorer.compute(hypotheses[i], references[i]):.4f}\n"
            report += f"METEOR:     {self.meteor_scorer.compute(hypotheses[i], references[i]):.4f}\n"
            report += "-" * 60 + "\n\n"
        
        return report


# Human evaluation template
def generate_human_evaluation_template(hypotheses: List[str], 
                                       references: List[str],
                                       source_texts: List[str],
                                       output_file: str = "human_eval.txt"):
    """
    Generate template for human evaluation
    
    Args:
        hypotheses: Translated sentences
        references: Reference translations
        source_texts: Source sentences
        output_file: Output file path
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("HUMAN EVALUATION TEMPLATE\n")
        f.write("=" * 80 + "\n\n")
        f.write("Please rate each translation on:\n")
        f.write("1. Fluency (1-5): How natural does the translation sound?\n")
        f.write("2. Adequacy (1-5): How much of the original meaning is preserved?\n")
        f.write("3. Cultural Appropriateness (1-5): How culturally appropriate is the translation?\n\n")
        f.write("=" * 80 + "\n\n")
        
        for i, (src, hyp, ref) in enumerate(zip(source_texts, hypotheses, references)):
            f.write(f"Sentence {i+1}:\n")
            f.write(f"Source:      {src}\n")
            f.write(f"Translation: {hyp}\n")
            f.write(f"Reference:   {ref}\n")
            f.write(f"Fluency:     [    ]\n")
            f.write(f"Adequacy:    [    ]\n")
            f.write(f"Cultural:    [    ]\n")
            f.write(f"Comments:    _____________________________________\n")
            f.write("-" * 80 + "\n\n")
    
    print(f"Human evaluation template saved to {output_file}")


if __name__ == "__main__":
    # Test evaluation metrics
    print("Testing Evaluation Metrics...\n")
    
    # Sample data
    hypotheses = [
        "The cat is on the mat",
        "A beautiful day in the park",
        "Machine translation is improving"
    ]
    
    references = [
        "The cat sits on the mat",
        "It's a beautiful day at the park",
        "Machine translation is getting better"
    ]
    
    source_texts = [
        "Die Katze ist auf der Matte",
        "Ein schöner Tag im Park",
        "Maschinelle Übersetzung wird besser"
    ]
    
    # Evaluate
    evaluator = TranslationEvaluator()
    
    # Generate report
    report = evaluator.generate_evaluation_report(hypotheses, references, source_texts)
    print(report)
    
    # Generate human evaluation template
    generate_human_evaluation_template(hypotheses, references, source_texts)
    
    print("\n✓ Evaluation metrics tested successfully!")
