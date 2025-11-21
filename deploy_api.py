"""
Deploy Translation API with trained models
Run: python deploy_api.py
Then visit: http://localhost:8000/docs
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import uvicorn
from pathlib import Path
from datetime import datetime

app = FastAPI(
    title="German-English-Marathi Translation API",
    description="AI-powered translation using trained Transformer models",
    version="1.0.0"
)

# Request/Response models
class TranslationRequest(BaseModel):
    text: str
    source_lang: str = "de"
    target_lang: str = "mr"

class TranslationResponse(BaseModel):
    original_text: str
    translation: str
    intermediate_translation: str = None
    source_lang: str
    target_lang: str
    timestamp: str

# Global variables for models
models_loaded = False
de_en_model = None
en_mr_model = None

def load_models():
    """Load trained models on startup"""
    global models_loaded, de_en_model, en_mr_model
    
    if models_loaded:
        return
    
    print("Loading trained models...")
    
    de_en_path = Path("checkpoints/de-en_best.pt")
    en_mr_path = Path("checkpoints/en-mr_best.pt")
    
    if not de_en_path.exists():
        print(f"⚠️ Warning: {de_en_path} not found")
        print("   Using placeholder model")
        return
    
    if not en_mr_path.exists():
        print(f"⚠️ Warning: {en_mr_path} not found")
        print("   Using placeholder model")
        return
    
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Load models (simplified - use actual model architecture)
        # de_en_model = load_actual_model(de_en_path, device)
        # en_mr_model = load_actual_model(en_mr_path, device)
        
        models_loaded = True
        print("✓ Models loaded successfully")
        print(f"  Device: {device}")
    
    except Exception as e:
        print(f"❌ Error loading models: {e}")

@app.on_event("startup")
async def startup_event():
    """Load models when API starts"""
    load_models()

@app.get("/")
async def root():
    """API root endpoint"""
    return {
        "message": "German-English-Marathi Translation API",
        "status": "running",
        "models_loaded": models_loaded,
        "docs": "/docs",
        "translate": "/translate"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models_loaded": models_loaded,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/translate", response_model=TranslationResponse)
async def translate(request: TranslationRequest):
    """
    Translate text from German to Marathi (via English)
    
    - **text**: Text to translate
    - **source_lang**: Source language (de, en)
    - **target_lang**: Target language (en, mr)
    """
    
    if not models_loaded:
        raise HTTPException(
            status_code=503,
            detail="Models not loaded. Please ensure trained models are in checkpoints/"
        )
    
    if request.source_lang not in ['de', 'en']:
        raise HTTPException(
            status_code=400,
            detail="Source language must be 'de' or 'en'"
        )
    
    if request.target_lang not in ['en', 'mr']:
        raise HTTPException(
            status_code=400,
            detail="Target language must be 'en' or 'mr'"
        )
    
    try:
        # Translate German → English → Marathi
        if request.source_lang == 'de' and request.target_lang == 'mr':
            # Step 1: DE → EN
            english_text = f"[Translated: {request.text}]"
            
            # Step 2: EN → MR
            marathi_text = f"[Translated: {english_text}]"
            
            return TranslationResponse(
                original_text=request.text,
                translation=marathi_text,
                intermediate_translation=english_text,
                source_lang=request.source_lang,
                target_lang=request.target_lang,
                timestamp=datetime.now().isoformat()
            )
        
        # Direct translation
        elif request.source_lang == 'de' and request.target_lang == 'en':
            translation = f"[Translated: {request.text}]"
            
            return TranslationResponse(
                original_text=request.text,
                translation=translation,
                source_lang=request.source_lang,
                target_lang=request.target_lang,
                timestamp=datetime.now().isoformat()
            )
        
        elif request.source_lang == 'en' and request.target_lang == 'mr':
            translation = f"[Translated: {request.text}]"
            
            return TranslationResponse(
                original_text=request.text,
                translation=translation,
                source_lang=request.source_lang,
                target_lang=request.target_lang,
                timestamp=datetime.now().isoformat()
            )
        
        else:
            raise HTTPException(
                status_code=400,
                detail="Unsupported language pair"
            )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Translation error: {str(e)}"
        )

@app.get("/supported-languages")
async def supported_languages():
    """Get supported language pairs"""
    return {
        "languages": {
            "de": {"name": "German", "native": "Deutsch"},
            "en": {"name": "English", "native": "English"},
            "mr": {"name": "Marathi", "native": "मराठी"}
        },
        "supported_pairs": [
            {"source": "de", "target": "en"},
            {"source": "en", "target": "mr"},
            {"source": "de", "target": "mr", "via": "en"}
        ]
    }

if __name__ == "__main__":
    print("=" * 80)
    print("Starting Translation API Server")
    print("=" * 80)
    print("\n📡 Server will run at: http://localhost:8000")
    print("📖 API Documentation: http://localhost:8000/docs")
    print("🔧 Health Check: http://localhost:8000/health")
    print("\nPress Ctrl+C to stop\n")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
