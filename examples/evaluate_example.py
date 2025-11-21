"""
Example script for evaluating translation models
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from evaluation.metrics import TranslationEvaluator, generate_human_evaluation_template
from api.api_server import TranslationPipeline

def evaluate_model_performance():
    """
    Evaluate model performance using BLEU and METEOR metrics
    """
    print("=" * 80)
    print("Model Evaluation")
    print("=" * 80)
    
    print("\n[1] Loading translation pipeline...")
    pipeline = TranslationPipeline()
    
    try:
        pipeline.load_models(
            de_en_path='checkpoints/de_en_best_model.pt',
            en_mr_path='checkpoints/en_mr_best_model.pt'
        )
        print("✓ Models loaded successfully!")
    except FileNotFoundError:
        print("⚠ Warning: Model checkpoints not found.")
        print("   Please train models first using examples/train_example.py")
        return
    
    print("\n[2] Loading test data...")
    
    # Load test datasets
    # TODO: Replace with actual test data
    german_test_sentences = [
        "Guten Morgen, wie geht es Ihnen?",
        "Ich liebe die deutsche Sprache.",
        "Die Wissenschaft ist sehr wichtig.",
        "Berlin ist die Hauptstadt von Deutschland.",
        "Danke für Ihre Hilfe."
    ]
    
    # Reference translations (ground truth)
    marathi_references = [
        "सुप्रभात, आपण कसे आहात?",
        "मला जर्मन भाषा आवडते.",
        "विज्ञान आपल्या भविष्यासाठी खूप महत्त्वाचे आहे.",
        "बर्लिन जर्मनीची राजधानी आहे.",
        "तुमच्या मदतीबद्दल धन्यवाद."
    ]
    
    english_references = [
        "Good morning, how are you?",
        "I love the German language.",
        "Science is very important for our future.",
        "Berlin is the capital of Germany.",
        "Thank you for your help."
    ]
    
    print(f"✓ Test set size: {len(german_test_sentences)} sentences")
    
    print("\n[3] Generating translations...")
    
    # Generate translations
    marathi_hypotheses = []
    english_hypotheses = []
    
    for german_text in german_test_sentences:
        try:
            marathi_text, english_text = pipeline.translate_de_to_mr(german_text)
            marathi_hypotheses.append(marathi_text)
            english_hypotheses.append(english_text)
        except Exception as e:
            print(f"   Error translating: {german_text}")
            print(f"   {e}")
            marathi_hypotheses.append("")
            english_hypotheses.append("")
    
    print("✓ Translations generated")
    
    print("\n[4] Calculating metrics...")
    
    # Evaluate German→English
    evaluator_de_en = TranslationEvaluator()
    results_de_en = evaluator_de_en.evaluate(english_hypotheses, english_references)
    
    print("\n📊 German → English Results:")
    print(f"   BLEU-1: {results_de_en['bleu_1']:.4f}")
    print(f"   BLEU-2: {results_de_en['bleu_2']:.4f}")
    print(f"   BLEU-3: {results_de_en['bleu_3']:.4f}")
    print(f"   BLEU-4: {results_de_en['bleu_4']:.4f}")
    print(f"   Corpus BLEU: {results_de_en['corpus_bleu']:.4f}")
    print(f"   Average METEOR: {results_de_en['avg_meteor']:.4f}")
    
    # Evaluate English→Marathi (via pipeline)
    evaluator_en_mr = TranslationEvaluator()
    results_en_mr = evaluator_en_mr.evaluate(marathi_hypotheses, marathi_references)
    
    print("\n📊 German → English → Marathi Results:")
    print(f"   BLEU-1: {results_en_mr['bleu_1']:.4f}")
    print(f"   BLEU-2: {results_en_mr['bleu_2']:.4f}")
    print(f"   BLEU-3: {results_en_mr['bleu_3']:.4f}")
    print(f"   BLEU-4: {results_en_mr['bleu_4']:.4f}")
    print(f"   Corpus BLEU: {results_en_mr['corpus_bleu']:.4f}")
    print(f"   Average METEOR: {results_en_mr['avg_meteor']:.4f}")
    
    print("\n[5] Generating detailed report...")
    
    # Generate evaluation report
    report_de_en = evaluator_de_en.generate_evaluation_report(
        english_hypotheses,
        english_references,
        source_texts=german_test_sentences
    )
    
    # Save report
    os.makedirs('evaluation_results', exist_ok=True)
    
    with open('evaluation_results/de_en_report.txt', 'w', encoding='utf-8') as f:
        f.write(report_de_en)
    
    report_en_mr = evaluator_en_mr.generate_evaluation_report(
        marathi_hypotheses,
        marathi_references,
        source_texts=german_test_sentences
    )
    
    with open('evaluation_results/de_en_mr_report.txt', 'w', encoding='utf-8') as f:
        f.write(report_en_mr)
    
    print("✓ Reports saved to evaluation_results/")
    
    print("\n[6] Sample translations:")
    print("-" * 80)
    
    for i in range(min(3, len(german_test_sentences))):
        print(f"\nExample {i+1}:")
        print(f"  🇩🇪 German:    {german_test_sentences[i]}")
        print(f"  🇬🇧 English:   {english_hypotheses[i]}")
        print(f"  🇬🇧 Reference: {english_references[i]}")
        print(f"  🇮🇳 Marathi:   {marathi_hypotheses[i]}")
        print(f"  🇮🇳 Reference: {marathi_references[i]}")
    
    print("\n" + "-" * 80)
    print("\n✓ Evaluation complete!")
    print("=" * 80)

def generate_human_eval_template():
    """
    Generate template for human evaluation
    """
    print("\n[7] Generating human evaluation template...")
    
    german_sentences = [
        "Guten Morgen, wie geht es Ihnen?",
        "Ich liebe die deutsche Sprache.",
        "Die Wissenschaft ist sehr wichtig.",
    ]
    
    marathi_translations = [
        "सुप्रभात, तुम्ही कसे आहात?",
        "मला जर्मन भाषा आवडते.",
        "विज्ञान खूप महत्त्वाचे आहे.",
    ]
    
    template = generate_human_evaluation_template(
        german_sentences,
        marathi_translations
    )
    
    os.makedirs('evaluation_results', exist_ok=True)
    
    with open('evaluation_results/human_evaluation_template.txt', 'w', encoding='utf-8') as f:
        f.write(template)
    
    print("✓ Human evaluation template saved to:")
    print("   evaluation_results/human_evaluation_template.txt")

def main():
    """
    Main evaluation pipeline
    """
    evaluate_model_performance()
    generate_human_eval_template()
    
    print("\n🎉 Evaluation complete!")
    print("   Check evaluation_results/ directory for detailed reports.")

if __name__ == "__main__":
    main()
