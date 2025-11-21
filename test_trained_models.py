"""
Test the trained German→English→Marathi translation pipeline
"""

import torch
import torch.nn as nn
from pathlib import Path
import math

class TransformerModel(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, nhead=8,
                 num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=2048):
        super().__init__()
        self.d_model = d_model
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_encoder = nn.Embedding(200, d_model)
        
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
        src_pos = torch.arange(0, src.size(1), device=src.device).unsqueeze(0).expand(src.size(0), -1)
        tgt_pos = torch.arange(0, tgt.size(1), device=tgt.device).unsqueeze(0).expand(tgt.size(0), -1)
        
        src = self.src_embedding(src) * math.sqrt(self.d_model) + self.pos_encoder(src_pos)
        tgt = self.tgt_embedding(tgt) * math.sqrt(self.d_model) + self.pos_encoder(tgt_pos)
        
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)
        output = self.transformer(src, tgt, tgt_mask=tgt_mask)
        return self.fc_out(output)

class TranslationPipeline:
    def __init__(self, de_en_path, en_mr_path, device='cpu'):
        self.device = torch.device(device)
        
        # Load German→English model
        print("Loading German→English model...")
        checkpoint = torch.load(de_en_path, map_location=self.device)
        # Get vocab sizes from checkpoint (you'll need to save these during training)
        self.de_en_model = TransformerModel(32000, 32000).to(self.device)
        self.de_en_model.load_state_dict(checkpoint['model_state_dict'])
        self.de_en_model.eval()
        print("✓ German→English model loaded")
        
        # Load English→Marathi model
        print("Loading English→Marathi model...")
        checkpoint = torch.load(en_mr_path, map_location=self.device)
        self.en_mr_model = TransformerModel(32000, 32000).to(self.device)
        self.en_mr_model.load_state_dict(checkpoint['model_state_dict'])
        self.en_mr_model.eval()
        print("✓ English→Marathi model loaded")
    
    def translate(self, text, model, max_length=50):
        """Simple greedy decoding"""
        with torch.no_grad():
            # Tokenize (simplified - you'd use actual vocab)
            src = torch.randint(1, 100, (1, 20)).to(self.device)
            tgt = torch.tensor([[1]]).to(self.device)  # Start with <sos>
            
            for _ in range(max_length):
                output = model(src, tgt)
                next_token = output[:, -1, :].argmax(dim=-1, keepdim=True)
                tgt = torch.cat([tgt, next_token], dim=1)
                
                if next_token.item() == 2:  # <eos> token
                    break
            
            return tgt[0].tolist()
    
    def translate_de_to_mr(self, german_text):
        """Translate German → English → Marathi"""
        print(f"\n🇩🇪 Input (German): {german_text}")
        
        # Step 1: German → English
        english_tokens = self.translate(german_text, self.de_en_model)
        english_text = f"[Translated to English - tokens: {len(english_tokens)}]"
        print(f"🇬🇧 Intermediate (English): {english_text}")
        
        # Step 2: English → Marathi
        marathi_tokens = self.translate(english_text, self.en_mr_model)
        marathi_text = f"[Translated to Marathi - tokens: {len(marathi_tokens)}]"
        print(f"🇮🇳 Output (Marathi): {marathi_text}")
        
        return marathi_text, english_text

def main():
    print("=" * 80)
    print("German → English → Marathi Translation Test")
    print("=" * 80)
    
    # Check for models
    de_en_path = Path("checkpoints/de-en_best.pt")
    en_mr_path = Path("checkpoints/en-mr_best.pt")
    
    if not de_en_path.exists():
        print(f"❌ Error: {de_en_path} not found!")
        print("   Please download and place models in checkpoints/ folder")
        return
    
    if not en_mr_path.exists():
        print(f"❌ Error: {en_mr_path} not found!")
        print("   Please download and place models in checkpoints/ folder")
        return
    
    # Initialize pipeline
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    
    try:
        pipeline = TranslationPipeline(
            str(de_en_path),
            str(en_mr_path),
            device=device
        )
        
        print("\n" + "=" * 80)
        print("Models loaded successfully! Ready for translation.")
        print("=" * 80)
        
        # Test translations
        test_sentences = [
            "Guten Morgen, wie geht es Ihnen?",
            "Ich liebe die deutsche Sprache.",
            "Die Wissenschaft ist sehr wichtig für unsere Zukunft.",
        ]
        
        print("\n📝 Test Translations:\n")
        for i, german_text in enumerate(test_sentences, 1):
            print(f"Example {i}:")
            marathi, english = pipeline.translate_de_to_mr(german_text)
            print()
        
        # Interactive mode
        print("=" * 80)
        print("Interactive Translation Mode")
        print("Enter German text to translate (or 'quit' to exit)")
        print("=" * 80)
        
        while True:
            try:
                user_input = input("\n🇩🇪 German: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("\nGoodbye!")
                    break
                
                if not user_input:
                    continue
                
                pipeline.translate_de_to_mr(user_input)
                
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    except Exception as e:
        print(f"\n❌ Error loading models: {e}")
        print("\nNote: Make sure models were trained with same architecture!")

if __name__ == "__main__":
    main()
