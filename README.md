# AI-Based Language Translation with Attention Mechanisms
## German → English → Marathi Translation System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/Transformers-4.30+-yellow.svg)](https://huggingface.co/transformers/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📋 Overview

This project implements a state-of-the-art Neural Machine Translation (NMT) system for translating from **German → English → Marathi** using Transformer models with attention mechanisms. The system leverages deep learning advances to provide accurate, contextually-aware multilingual translation, particularly focusing on the low-resource language pair German-Marathi.

### Key Features

- ✅ **Transformer Architecture**: Self-attention and multi-head attention mechanisms
- ✅ **Pipeline Translation**: German → English → Marathi with intermediate outputs
- ✅ **Attention Mechanisms**: Context-aware word-level alignment
- ✅ **Low-Resource Support**: Optimized for Marathi translation
- ✅ **REST API**: Real-time translation endpoint
- ✅ **Evaluation Metrics**: BLEU and METEOR scoring
- ✅ **Production Ready**: Scalable deployment with FastAPI

## 🏗️ Architecture

The system consists of two sequential Transformer models:

```
German Text → [Encoder-Decoder 1] → English Text → [Encoder-Decoder 2] → Marathi Text
                  (DE→EN Model)                        (EN→MR Model)
```

### Transformer Components

1. **Encoder**: Processes input sequence with self-attention
2. **Decoder**: Generates output sequence with cross-attention
3. **Multi-Head Attention**: Learns multiple semantic relationships
4. **Positional Encoding**: Maintains sequence order information
5. **Feed-Forward Networks**: Non-linear transformations

![Transformer Architecture](docs/architecture.png)

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (optional, but recommended)
- 8GB+ RAM

### Installation

```bash
# Clone the repository
git clone https://github.com/mohittade/ai_gr_to.git
cd ai_gr_to

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

#### 1. Data Preprocessing

```python
from utils.preprocessing import DataPreprocessor, BilingualDataset

# Initialize preprocessor
preprocessor = DataPreprocessor(config={
    'max_length': 128,
    'min_length': 3
})

# Load and preprocess data
dataset = BilingualDataset(
    source_file='data/german.txt',
    target_file='data/english.txt',
    preprocessor=preprocessor
)
```

#### 2. Training Models

```python
from training.train import Trainer, Transformer
from torch.utils.data import DataLoader

# Configuration
config = {
    'd_model': 512,
    'num_heads': 8,
    'num_encoder_layers': 6,
    'num_decoder_layers': 6,
    'd_ff': 2048,
    'dropout': 0.1,
    'batch_size': 32,
    'num_epochs': 30
}

# Create model
model = Transformer(
    src_vocab_size=32000,
    tgt_vocab_size=32000,
    **config
)

# Train
trainer = Trainer(model, train_loader, val_loader, config)
trainer.train(num_epochs=30)
```

#### 3. Translation

```python
from api.api_server import TranslationPipeline

# Initialize pipeline
pipeline = TranslationPipeline()
pipeline.load_models(
    de_en_path='models/de_en_model.pt',
    en_mr_path='models/en_mr_model.pt'
)

# Translate
german_text = "Guten Morgen, wie geht es Ihnen?"
marathi_text, english_intermediate = pipeline.translate_de_to_mr(german_text)

print(f"German: {german_text}")
print(f"English (intermediate): {english_intermediate}")
print(f"Marathi: {marathi_text}")
```

#### 4. Evaluation

```python
from evaluation.metrics import TranslationEvaluator

evaluator = TranslationEvaluator()

hypotheses = ["translated sentence 1", "translated sentence 2"]
references = ["reference translation 1", "reference translation 2"]

# Evaluate
results = evaluator.evaluate(hypotheses, references)
print(f"BLEU Score: {results['corpus_bleu']:.4f}")
print(f"METEOR Score: {results['avg_meteor']:.4f}")

# Generate report
report = evaluator.generate_evaluation_report(hypotheses, references)
print(report)
```

### REST API Usage

#### Start the API Server

```bash
# Run the FastAPI server
python api/api_server.py

# Or using uvicorn directly
uvicorn api.api_server:app --host 0.0.0.0 --port 8000 --reload
```

#### API Endpoints

**Translate Text**
```bash
curl -X POST "http://localhost:8000/translate" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Guten Morgen",
    "source_lang": "de",
    "target_lang": "mr"
  }'
```

**Batch Translation**
```bash
curl -X POST "http://localhost:8000/batch-translate" \
  -H "Content-Type: application/json" \
  -d '{
    "texts": ["Hallo", "Danke"],
    "source_lang": "de",
    "target_lang": "mr"
  }'
```

**Health Check**
```bash
curl http://localhost:8000/health
```

**Interactive Documentation**
Visit `http://localhost:8000/docs` for interactive API documentation powered by Swagger UI.

## 📊 Model Performance

| Metric | DE→EN | EN→MR | DE→MR (Pipeline) |
|--------|-------|-------|------------------|
| BLEU-4 | 0.XX  | 0.XX  | 0.XX            |
| METEOR | 0.XX  | 0.XX  | 0.XX            |

*Note: Scores will be updated after training on full datasets*

## 📁 Project Structure

```
ai_gr_to/
├── data/                   # Training and test datasets
│   ├── german_english/     # German-English parallel corpus
│   └── english_marathi/    # English-Marathi parallel corpus
├── models/                 # Model implementations
│   └── transformer.py      # Transformer architecture
├── training/               # Training scripts
│   └── train.py           # Training pipeline
├── evaluation/            # Evaluation metrics
│   └── metrics.py         # BLEU, METEOR implementation
├── api/                   # REST API
│   └── api_server.py      # FastAPI server
├── utils/                 # Utilities
│   └── preprocessing.py   # Data preprocessing
├── checkpoints/           # Model checkpoints
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## 🔬 Technical Details

### Attention Mechanism

The multi-head attention mechanism allows the model to focus on different parts of the input sequence:

```python
Attention(Q, K, V) = softmax(QK^T / √d_k)V
```

Where:
- Q (Query): What we're looking for
- K (Key): What we're looking at
- V (Value): The actual information
- d_k: Dimension scaling factor

### Preprocessing Pipeline

1. **Text Cleaning**: Remove URLs, emails, special characters
2. **Normalization**: Unicode normalization, lowercasing, punctuation standardization
3. **Tokenization**: Byte Pair Encoding (BPE) for subword segmentation
4. **Alignment**: Validate parallel sentence pairs

### Training Strategy

- **Optimizer**: Adam with β₁=0.9, β₂=0.98, ε=1e-9
- **Learning Rate**: Warmup for 4000 steps, then decay
- **Regularization**: Label smoothing (0.1), dropout (0.1)
- **Batch Size**: Dynamic batching for GPU utilization
- **Loss Function**: Cross-entropy with label smoothing

## 📚 Datasets

### German-English
- **Europarl**: European Parliament proceedings
- **OPUS**: Open parallel corpus
- **Size**: ~1M sentence pairs

### English-Marathi
- **IIT Bombay Corpus**: English-Hindi-Marathi
- **OPUS**: Various sources
- **Size**: ~50K sentence pairs

## 🎯 Use Cases

- **Education**: Translate German academic content to Marathi
- **Healthcare**: Medical document translation
- **Business**: Cross-cultural communication
- **Research**: Multilingual knowledge sharing
- **Digital Content**: Website and app localization

## 🛠️ Development

### Running Tests

```bash
# Test preprocessing
python utils/preprocessing.py

# Test model architecture
python models/transformer.py

# Test evaluation metrics
python evaluation/metrics.py
```

### Training from Scratch

```bash
# 1. Prepare data
python utils/preprocessing.py --input data/raw --output data/processed

# 2. Train German→English model
python training/train.py --config configs/de_en_config.json

# 3. Train English→Marathi model
python training/train.py --config configs/en_mr_config.json

# 4. Evaluate
python evaluation/metrics.py --model checkpoints/best_model.pt
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📖 Citation

If you use this system in your research, please cite:

```bibtex
@article{raina2024german_marathi_translation,
  title={AI-Based Language Translation with Attention Mechanisms: German to English to Marathi},
  author={Raina, Ricky and Londhe, Aryan and Dhole, Aditi and Salve, Shreya},
  journal={Information Technology Department, JSPM's Rajarshi Shahu College of Engineering},
  year={2024}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- **Ricky Raina** - [Email](mailto:Rickyraina11@gmail.com)
- **Aryan Londhe** - [Email](mailto:londhearyan21@gmail.com)
- **Aditi Dhole** - [Email](mailto:dholeaditi56@gmail.com)
- **Shreya Salve** - [Email](mailto:Salveshreya.official@gmail.com)

**Supervisor**: Dr. Archana Jadhav

**Institution**: JSPM's Rajarshi Shahu College of Engineering, Tathawade, Pune, India

## 🙏 Acknowledgments

- Based on "Attention Is All You Need" (Vaswani et al., 2017)
- HuggingFace Transformers library
- PyTorch team
- OPUS and Europarl dataset contributors
- IIT Bombay for English-Marathi corpus

## 📞 Contact

For questions or collaborations, please contact:
- Email: Rickyraina11@gmail.com
- GitHub: [@mohittade](https://github.com/mohittade)

---

**Status**: 🟢 Active Development | **Version**: 1.0.0 | **Last Updated**: November 2024
