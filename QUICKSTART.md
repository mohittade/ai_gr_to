# German → English → Marathi Translation System

## Quick Start Guide

This guide will help you get the translation system up and running quickly.

## Installation

### 1. Prerequisites
- Python 3.8 or higher
- Git
- (Optional) CUDA-capable GPU for training

### 2. Clone and Setup

```powershell
# Clone repository
git clone https://github.com/mohittade/ai_gr_to.git
cd ai_gr_to

# Run automated setup
.\setup.ps1

# Or manual setup:
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Usage Examples

### 1. Basic Translation

```python
from api.api_server import TranslationPipeline

# Initialize
pipeline = TranslationPipeline()
pipeline.load_models(
    de_en_path='checkpoints/de_en_best_model.pt',
    en_mr_path='checkpoints/en_mr_best_model.pt'
)

# Translate
german = "Guten Morgen, wie geht es Ihnen?"
marathi, english = pipeline.translate_de_to_mr(german)

print(f"German:  {german}")
print(f"English: {english}")
print(f"Marathi: {marathi}")
```

### 2. Using the API

Start the server:
```powershell
python api/api_server.py
```

Send translation requests:
```powershell
# PowerShell
$body = @{
    text = "Guten Morgen"
    source_lang = "de"
    target_lang = "mr"
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:8000/translate" `
    -Method Post `
    -ContentType "application/json" `
    -Body $body
```

### 3. Training Models

```powershell
# Train both models
python examples/train_example.py --model both

# Train only German-English
python examples/train_example.py --model de-en

# Train only English-Marathi
python examples/train_example.py --model en-mr
```

### 4. Evaluation

```powershell
# Evaluate models
python examples/evaluate_example.py
```

## Project Structure

```
ai_gr_to/
├── api/                    # REST API
│   └── api_server.py
├── data/                   # Training data
│   ├── german_english/
│   └── english_marathi/
├── models/                 # Model architecture
│   └── transformer.py
├── training/               # Training scripts
│   └── train.py
├── evaluation/            # Evaluation metrics
│   └── metrics.py
├── utils/                 # Utilities
│   └── preprocessing.py
├── examples/              # Example scripts
│   ├── basic_translation.py
│   ├── train_example.py
│   ├── evaluate_example.py
│   └── api_client_example.py
├── checkpoints/           # Saved models
├── config.json           # Configuration
└── requirements.txt      # Dependencies
```

## Configuration

Edit `config.json` to customize:
- Model architecture (dimensions, layers, heads)
- Training parameters (batch size, learning rate)
- API settings (port, workers)
- Data paths

## Common Tasks

### Change Model Size
```json
// config.json
{
  "model": {
    "d_model": 256,        // Smaller model
    "num_heads": 4,
    "num_encoder_layers": 3,
    "num_decoder_layers": 3
  }
}
```

### Adjust Training
```json
// config.json
{
  "training": {
    "batch_size": 64,      // Larger batches
    "num_epochs": 50,      // More epochs
    "learning_rate": 0.0005
  }
}
```

### API Configuration
```json
// config.json
{
  "api": {
    "host": "0.0.0.0",
    "port": 8080,          // Different port
    "workers": 8           // More workers
  }
}
```

## Troubleshooting

### Out of Memory
- Reduce batch size in `config.json`
- Use gradient accumulation
- Use smaller model dimensions

### Slow Training
- Enable CUDA if available
- Increase batch size
- Use mixed precision training

### API Not Responding
- Check if server is running: `curl http://localhost:8000/health`
- Check logs in `logs/` directory
- Verify models are loaded correctly

## Performance Tips

1. **GPU Usage**: Train on GPU for 10-50x speedup
2. **Batch Size**: Larger batches = faster training (if memory allows)
3. **Data Quality**: Clean, aligned data = better translations
4. **Model Size**: Larger models = better quality (but slower)

## Next Steps

1. **Prepare Data**: Download and prepare parallel corpora
2. **Train Models**: Start with small datasets to verify setup
3. **Evaluate**: Use BLEU/METEOR metrics to assess quality
4. **Deploy**: Use FastAPI for production deployment
5. **Iterate**: Fine-tune hyperparameters based on results

## Support

- Documentation: See `README.md`
- Examples: Check `examples/` directory
- Issues: [GitHub Issues](https://github.com/mohittade/ai_gr_to/issues)

## Resources

- [Attention Is All You Need (Vaswani et al.)](https://arxiv.org/abs/1706.03762)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [BLEU Score Explanation](https://en.wikipedia.org/wiki/BLEU)
