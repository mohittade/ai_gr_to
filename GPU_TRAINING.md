# German-English-Marathi Translation Training on Google Colab

## Quick Setup Guide

### Step 1: Open Google Colab
1. Go to [https://colab.research.google.com/](https://colab.research.google.com/)
2. Sign in with your Google account
3. Click **File → New Notebook**

### Step 2: Enable GPU
1. Click **Runtime → Change runtime type**
2. Select **Hardware accelerator: GPU (T4 or V100)**
3. Click **Save**

### Step 3: Upload Your Data
Run this in a Colab cell:
```python
from google.colab import drive
drive.mount('/content/drive')
```

Or upload directly:
```python
from google.colab import files
uploaded = files.upload()  # Upload your zip file
!unzip your_data.zip
```

### Step 4: Install Dependencies
```python
!pip install transformers sentencepiece fastapi uvicorn sacrebleu nltk tqdm
```

### Step 5: Clone Your Repository
```python
!git clone https://github.com/mohittade/ai_gr_to.git
%cd ai_gr_to
```

### Step 6: Upload Your Data Files
```python
# Upload to Colab
from google.colab import files
import shutil

# Upload German-English data
print("Upload europarl.de file:")
uploaded = files.upload()
shutil.move(list(uploaded.keys())[0], 'data/german_english/europarl.de')

print("Upload europarl.en file:")
uploaded = files.upload()
shutil.move(list(uploaded.keys())[0], 'data/german_english/europarl.en')

# Upload English-Marathi data
print("Upload train.en file:")
uploaded = files.upload()
shutil.move(list(uploaded.keys())[0], 'data/english_marathi/train.en')

print("Upload train.mr file:")
uploaded = files.upload()
shutil.move(list(uploaded.keys())[0], 'data/english_marathi/train.mr')
```

### Step 7: Prepare Data
```python
!python prepare_data.py
!python prepare_en_mr_data.py
```

### Step 8: Train Models
```python
# Train German-English (will take ~6-12 hours on T4 GPU)
!python examples/train_example.py --model de-en

# Train English-Marathi (will take ~2-4 hours on T4 GPU)
!python examples/train_example.py --model en-mr
```

### Step 9: Download Trained Models
```python
from google.colab import files

# Download checkpoints
files.download('checkpoints/de_en_best_model.pt')
files.download('checkpoints/en_mr_best_model.pt')
```

## Tips for Colab Training

### 1. Session Time Limits
- Free Colab: 12 hours max
- Colab Pro: 24 hours max
- Save checkpoints frequently!

### 2. Prevent Disconnection
```python
# Run this to keep session alive
import time
from IPython.display import clear_output

while True:
    time.sleep(60)
    clear_output(wait=True)
    print("Session active...")
```

### 3. Monitor GPU Usage
```python
!nvidia-smi
```

### 4. Save to Google Drive
```python
# Mount Drive
from google.colab import drive
drive.mount('/content/drive')

# Save checkpoints to Drive
import shutil
shutil.copy('checkpoints/de_en_best_model.pt', '/content/drive/MyDrive/ai_models/')
```

## Alternative: Kaggle Notebooks

Kaggle also offers free GPUs (30 hours/week):

1. Go to [https://www.kaggle.com/](https://www.kaggle.com/)
2. Create a new Notebook
3. Turn on GPU in Settings
4. Upload data as Kaggle Dataset
5. Run training code

## Speed Comparison

| Hardware | DE→EN Training Time | EN→MR Training Time |
|----------|---------------------|---------------------|
| CPU (yours) | ~5-7 days | ~2-3 days |
| Google Colab T4 | ~8-12 hours | ~3-5 hours |
| Google Colab V100 | ~4-6 hours | ~1-2 hours |
| RTX 5060* | ~3-5 hours | ~1-2 hours |

*When PyTorch adds support

## Cost Options

| Service | GPU | Cost | Time Limit |
|---------|-----|------|------------|
| Google Colab Free | T4 | Free | 12 hours/session |
| Google Colab Pro | T4/V100 | $10/month | 24 hours/session |
| Kaggle | P100 | Free | 30 hours/week |
| AWS EC2 (g4dn.xlarge) | T4 | ~$0.50/hour | Unlimited |
| Paperspace Gradient | Various | $0.45/hour | Unlimited |
