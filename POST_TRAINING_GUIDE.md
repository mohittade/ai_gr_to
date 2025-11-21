# Post-Training Workflow

After downloading your trained models from Google Colab, follow these steps:

## Step 1: Place Models

Download `de-en_best.pt` and `en-mr_best.pt` from Colab and place them in the `checkpoints/` folder:

```
checkpoints/
├── de-en_best.pt      # German-English model (~230 MB)
└── en-mr_best.pt      # English-Marathi model (~230 MB)
```

## Step 2: Test Translations

Run the test script to verify models work:

```bash
python test_trained_models.py
```

This will:
- Load both trained models
- Run 3 test translations
- Enter interactive mode for you to test custom sentences
- Show intermediate English translations

Example output:
```
✓ Loaded DE-EN model (60.2M parameters)
✓ Loaded EN-MR model (59.8M parameters)

Test 1: "Guten Morgen"
  → English: "Good morning"
  → Marathi: "शुभ प्रभात"

Enter German text (or 'quit'): Das Wetter ist schön heute
  → English: "The weather is nice today"
  → Marathi: "आज हवामान छान आहे"
```

## Step 3: Evaluate Quality

Run the evaluation script to check BLEU scores:

```bash
python evaluate_trained_models.py
```

This will:
- Load test datasets (100 samples each)
- Calculate BLEU and METEOR scores
- Compare against reference translations

Expected scores:
- **DE-EN BLEU**: 28-35 (good commercial quality)
- **EN-MR BLEU**: 18-25 (good for low-resource pair)
- **Pipeline BLEU**: 15-20 (acceptable cascaded translation)

## Step 4: Deploy API

Start the REST API server:

```bash
python deploy_api.py
```

Then visit:
- **Interactive Docs**: http://localhost:8000/docs
- **API Root**: http://localhost:8000
- **Health Check**: http://localhost:8000/health

### Test API with curl:

```bash
curl -X POST "http://localhost:8000/translate" \
  -H "Content-Type: application/json" \
  -d '{"text": "Guten Morgen", "source_lang": "de", "target_lang": "mr"}'
```

Response:
```json
{
  "original_text": "Guten Morgen",
  "translation": "शुभ प्रभात",
  "intermediate_translation": "Good morning",
  "source_lang": "de",
  "target_lang": "mr",
  "timestamp": "2025-01-21T10:30:00"
}
```

## Step 5: Production Deployment

For production, consider:

### Option A: Docker Container

```bash
# Build image
docker build -t translation-api .

# Run container
docker run -p 8000:8000 translation-api
```

### Option B: Cloud Hosting

- **Heroku**: Free tier available, easy deployment
- **AWS Lambda**: Serverless, pay per request
- **Google Cloud Run**: Auto-scaling, containerized
- **Azure App Service**: Managed hosting

### Option C: Local Server

```bash
# Install as systemd service (Linux)
sudo cp translation-api.service /etc/systemd/system/
sudo systemctl enable translation-api
sudo systemctl start translation-api
```

## Quality Expectations

Based on your training data:

### German-English Translation
- **Training data**: 1.88M sentence pairs (Europarl)
- **Expected BLEU**: 28-35
- **Quality**: Suitable for professional use
- **Strengths**: Political/formal text, European languages
- **Weaknesses**: Idioms, slang, technical jargon

### English-Marathi Translation
- **Training data**: 113K sentence pairs (government press releases)
- **Expected BLEU**: 18-25
- **Quality**: Good for formal/official content
- **Strengths**: Government terminology, formal writing
- **Weaknesses**: Casual conversation, technical domains

### Full Pipeline (German → Marathi)
- **Expected BLEU**: 15-20
- **Quality**: Acceptable for gist understanding
- **Cascading loss**: ~40% from two-stage translation
- **Best for**: General understanding, not critical translations

## Troubleshooting

### Models not loading?

Check that files exist:
```bash
ls -lh checkpoints/
# Should show:
# de-en_best.pt (~230 MB)
# en-mr_best.pt (~230 MB)
```

### Out of memory?

Reduce batch size in API:
```python
# In deploy_api.py
max_batch_size = 1  # Process one at a time
```

Or use CPU instead of GPU:
```python
device = 'cpu'  # Slower but more memory
```

### Poor translation quality?

1. Check BLEU scores with `evaluate_trained_models.py`
2. If BLEU < 20 for DE-EN, model undertrained
3. If BLEU < 15 for EN-MR, model undertrained
4. Consider training longer (30-40 epochs)

### API errors?

Check logs:
```bash
# Run with verbose logging
python deploy_api.py --log-level debug
```

Test endpoints:
```bash
# Health check
curl http://localhost:8000/health

# Supported languages
curl http://localhost:8000/supported-languages
```

## Next Steps

1. **Integrate with applications**: Use the REST API in your web/mobile apps
2. **Fine-tune models**: Train longer on domain-specific data
3. **Add more languages**: Extend to other language pairs
4. **Optimize inference**: Use beam search, caching, batching
5. **Monitor performance**: Log translations, collect user feedback

---

## Support

If you encounter issues:

1. Check that models are fully trained (not quick_train demo models)
2. Verify BLEU scores match expected ranges
3. Test with simple sentences first
4. Review training logs in Colab
5. Ensure all dependencies installed (`pip install -r requirements.txt`)

**Demo models (from quick_train.py) will NOT work for production.** You must train full models in Colab first.
