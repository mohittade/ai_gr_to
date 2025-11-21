"""
Basic translation example using the German→English→Marathi pipeline
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.api_server import TranslationPipeline

def main():
    print("=" * 80)
    print("German → English → Marathi Translation Example")
    print("=" * 80)
    
    # Initialize translation pipeline
    print("\n[1] Loading translation models...")
    pipeline = TranslationPipeline()
    
    # Load trained models (you need to train these first)
    try:
        pipeline.load_models(
            de_en_path='checkpoints/de_en_best_model.pt',
            en_mr_path='checkpoints/en_mr_best_model.pt'
        )
        print("✓ Models loaded successfully!")
    except FileNotFoundError:
        print("⚠ Warning: Model checkpoints not found. Please train models first.")
        print("   Using placeholder models for demonstration...")
    
    # Example German sentences
    german_sentences = [
        "Guten Morgen, wie geht es Ihnen?",
        "Ich liebe die deutsche Sprache.",
        "Die Wissenschaft ist sehr wichtig für unsere Zukunft.",
        "Berlin ist die Hauptstadt von Deutschland.",
        "Danke für Ihre Hilfe."
    ]
    
    print("\n[2] Translating German sentences to Marathi...\n")
    print("-" * 80)
    
    for i, german_text in enumerate(german_sentences, 1):
        print(f"\nExample {i}:")
        print(f"  🇩🇪 German:  {german_text}")
        
        # Translate
        try:
            marathi_text, english_text = pipeline.translate_de_to_mr(german_text)
            print(f"  🇬🇧 English: {english_text}")
            print(f"  🇮🇳 Marathi: {marathi_text}")
        except Exception as e:
            print(f"  ❌ Error: {e}")
            print(f"  🇬🇧 English: [Translation would appear here]")
            print(f"  🇮🇳 Marathi: [Translation would appear here]")
    
    print("\n" + "-" * 80)
    
    # Interactive mode
    print("\n[3] Interactive Translation Mode")
    print("Enter German text to translate (or 'quit' to exit):\n")
    
    while True:
        try:
            user_input = input("🇩🇪 German: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not user_input:
                continue
            
            marathi_text, english_text = pipeline.translate_de_to_mr(user_input)
            print(f"🇬🇧 English: {english_text}")
            print(f"🇮🇳 Marathi: {marathi_text}\n")
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}\n")

if __name__ == "__main__":
    main()
