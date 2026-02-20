import fastapi
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
from PIL import Image
import io
import os
import json
import time
import logging
from pathlib import Path
from typing import List, Optional
import asyncio
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Try to import the classifier, handle if not available
try:
    from inference import PestClassifier
    INFERENCE_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Inference module not available: {e}")
    INFERENCE_AVAILABLE = False

app = FastAPI(
    title="AI Pest Detection API",
    description="Detect pests in crop images using deep learning",
    version="2.0.0",
    contact={
        "name": "Roy Kiprop Kongwalei",
        "email": "royk@brandeis.edu",
    }
)

# Enable CORS with more specific options
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
MODEL_PATH = "models/pest_classifier.tflite"
LABELS_PATH = "models/labels.json"
UPLOAD_DIR = "uploads"
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}

# Create necessary directories
os.makedirs("models", exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Initialize classifier with retry logic
classifier = None
use_mock = False

def initialize_classifier():
    """Initialize classifier with retry logic"""
    global classifier, use_mock
    
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        logger.warning(f"Model not found at {MODEL_PATH}")
        
        # Check for alternative model formats
        alternative_models = [
            "models/final_model.keras",
            "models/best_model.keras",
            "models/pest_model.h5",
            "models/model.tflite"
        ]
        
        for alt_model in alternative_models:
            if os.path.exists(alt_model):
                logger.info(f"Found alternative model: {alt_model}")
                # You might need to convert this to TFLite
                return setup_mock_classifier()
        
        return setup_mock_classifier()
    
    # Check if labels exist
    if not os.path.exists(LABELS_PATH):
        logger.warning(f"Labels not found at {LABELS_PATH}")
        return setup_mock_classifier()
    
    # Try to load the model
    try:
        if INFERENCE_AVAILABLE:
            classifier = PestClassifier(MODEL_PATH, LABELS_PATH)
            logger.info("✅ Model loaded successfully")
            use_mock = False
            return classifier
        else:
            logger.warning("Inference module not available")
            return setup_mock_classifier()
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return setup_mock_classifier()

def setup_mock_classifier():
    """Setup mock classifier for testing"""
    global use_mock
    use_mock = True
    
    # Default pest classes
    default_pests = [
        "aphids", "beetles", "caterpillars", "grasshoppers", 
        "spider_mites", "whiteflies", "thrips", "leafminers",
        "cutworms", "armyworms", "stemborers", "fruitflies"
    ]
    
    # Create mock labels file if it doesn't exist
    if not os.path.exists(LABELS_PATH):
        with open(LABELS_PATH, 'w') as f:
            json.dump(default_pests[:5], f)
    
    class MockClassifier:
        def __init__(self):
            with open(LABELS_PATH, 'r') as f:
                self.labels = json.load(f)
            logger.info(f"Mock classifier initialized with {len(self.labels)} classes")
        
        def predict(self, image, top_k=3):
            import random
            # Add some intelligence to mock predictions based on image properties
            img_array = np.array(image)
            
            # Use image properties to influence predictions
            brightness = np.mean(img_array)
            color_variance = np.std(img_array)
            
            # Generate more realistic mock predictions
            predictions = []
            used_indices = set()
            
            # Ensure top prediction is somewhat consistent for same image
            random.seed(int(brightness * 1000))
            
            for _ in range(min(top_k, len(self.labels))):
                available = [i for i in range(len(self.labels)) if i not in used_indices]
                if not available:
                    break
                    
                idx = random.choice(available)
                used_indices.add(idx)
                
                # Confidence varies based on image quality
                confidence = min(0.95, 0.6 + (color_variance / 255) * 0.3)
                confidence += random.uniform(-0.1, 0.1)
                confidence = max(0.5, min(0.99, confidence))
                
                predictions.append({
                    'class': self.labels[idx],
                    'confidence': round(confidence, 3)
                })
            
            return sorted(predictions, key=lambda x: x['confidence'], reverse=True)
    
    return MockClassifier()

# Initialize classifier on startup
@app.on_event("startup")
async def startup_event():
    """Run on application startup"""
    global classifier
    logger.info("🚀 Starting AI Pest Detection API")
    classifier = initialize_classifier()
    
    # Log configuration
    logger.info(f"📁 Model path: {MODEL_PATH}")
    logger.info(f"📁 Labels path: {LABELS_PATH}")
    logger.info(f"📁 Upload directory: {UPLOAD_DIR}")
    logger.info(f"📊 Max file size: {MAX_FILE_SIZE / (1024*1024)}MB")
    logger.info(f"🎯 Mode: {'MOCK' if use_mock else 'PRODUCTION'}")

@app.on_event("shutdown")
async def shutdown_event():
    """Run on application shutdown"""
    logger.info("👋 Shutting down API")

# Helper functions
def validate_image(file: UploadFile) -> tuple[bool, str]:
    """Validate uploaded image"""
    # Check file extension
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        return False, f"File type {ext} not allowed. Allowed: {ALLOWED_EXTENSIONS}"
    
    return True, ""

def get_treatment_info(pest_class: str) -> dict:
    """Get treatment information for a pest"""
    treatments = {
        "aphids": {
            "organic": "🌱 Spray neem oil or insecticidal soap. Introduce ladybugs.",
            "chemical": "🧪 Apply imidacloprid or pyrethroids.",
            "prevention": "🛡️ Use reflective mulches and avoid over-fertilizing.",
            "severity": "Medium",
            "action": "Immediate treatment recommended"
        },
        "beetles": {
            "organic": "🌱 Handpick beetles. Use row covers.",
            "chemical": "🧪 Apply carbaryl or spinosad.",
            "prevention": "🛡️ Rotate crops and maintain garden hygiene.",
            "severity": "Medium",
            "action": "Monitor regularly"
        },
        "caterpillars": {
            "organic": "🌱 Use Bacillus thuringiensis (Bt). Handpick.",
            "chemical": "🧪 Apply chlorantraniliprole.",
            "prevention": "🛡️ Use pheromone traps.",
            "severity": "High",
            "action": "Immediate action required"
        },
        "grasshoppers": {
            "organic": "🌱 Apply neem oil. Use nosema locustae.",
            "chemical": "🧪 Use carbaryl bait.",
            "prevention": "🛡️ Keep vegetation short around fields.",
            "severity": "Medium",
            "action": "Control before population explodes"
        },
        "spider_mites": {
            "organic": "🌱 Spray water to dislodge. Use predatory mites.",
            "chemical": "🧪 Apply miticides like abamectin.",
            "prevention": "🛡️ Maintain humidity and avoid drought stress.",
            "severity": "High",
            "action": "Treat immediately"
        },
        "whiteflies": {
            "organic": "🌱 Use yellow sticky traps. Introduce parasitic wasps.",
            "chemical": "🧪 Apply insecticidal soap or pyrethrins.",
            "prevention": "🛡️ Use reflective mulch. Avoid over-fertilizing.",
            "severity": "Medium",
            "action": "Monitor and treat at first sign"
        },
        "thrips": {
            "organic": "🌱 Use blue sticky traps. Introduce predatory mites.",
            "chemical": "🧪 Apply spinosad or neem oil.",
            "prevention": "🛡️ Remove weed hosts. Use reflective mulch.",
            "severity": "Medium",
            "action": "Quick action needed"
        },
        "leafminers": {
            "organic": "🌱 Remove affected leaves. Use parasitic wasps.",
            "chemical": "🧪 Apply spinosad or abamectin.",
            "prevention": "🛡️ Use row covers. Rotate crops.",
            "severity": "Low",
            "action": "Usually cosmetic damage only"
        }
    }
    
    return treatments.get(pest_class.lower(), {
        "organic": "🌱 Consult local agricultural extension office.",
        "chemical": "🧪 Consult with pest control professional.",
        "prevention": "🛡️ Regular crop monitoring and good farming practices.",
        "severity": "Unknown",
        "action": "Consult expert"
    })

# API Endpoints
@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint with API info"""
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>AI Pest Detection API</title>
        <style>
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                max-width: 800px;
                margin: 50px auto;
                padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                line-height: 1.6;
            }
            .container {
                background: rgba(255, 255, 255, 0.1);
                backdrop-filter: blur(10px);
                padding: 40px;
                border-radius: 20px;
                box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
            }
            h1 { margin-top: 0; }
            .endpoint {
                background: rgba(255, 255, 255, 0.2);
                padding: 15px;
                border-radius: 10px;
                margin: 10px 0;
            }
            code {
                background: rgba(0, 0, 0, 0.3);
                padding: 3px 6px;
                border-radius: 5px;
            }
            .status {
                display: inline-block;
                padding: 5px 10px;
                border-radius: 5px;
                background: #10b981;
                margin-top: 10px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🌱 AI Pest Detection API</h1>
            <p>Version 2.0.0 | FastAPI + TensorFlow Lite</p>
            
            <div class="status">🟢 API is running</div>
            
            <h2>📡 Available Endpoints</h2>
            
            <div class="endpoint">
                <strong>GET /</strong> - API information (this page)
            </div>
            
            <div class="endpoint">
                <strong>GET /health</strong> - Health check
                <br>
                <code>curl {os.getenv('HOST', 'http://localhost:8000')}/health</code>
            </div>
            
            <div class="endpoint">
                <strong>GET /classes</strong> - List all pest classes
                <br>
                <code>curl {os.getenv('HOST', 'http://localhost:8000')}/classes</code>
            </div>
            
            <div class="endpoint">
                <strong>POST /predict</strong> - Upload image for pest detection
                <br>
                <code>curl -X POST -F "file=@pest_image.jpg" {os.getenv('HOST', 'http://localhost:8000')}/predict</code>
            </div>
            
            <div class="endpoint">
                <strong>POST /predict/batch</strong> - Upload multiple images
                <br>
                <code>curl -X POST -F "files=@image1.jpg" -F "files=@image2.jpg" {os.getenv('HOST', 'http://localhost:8000')}/predict/batch</code>
            </div>
            
            <div class="endpoint">
                <strong>GET /docs</strong> - Interactive API documentation
                <br>
                <a href="/docs" style="color: white;">Open Swagger UI</a>
            </div>
            
            <div class="endpoint">
                <strong>GET /redoc</strong> - ReDoc documentation
                <br>
                <a href="/redoc" style="color: white;">Open ReDoc</a>
            </div>
            
            <h2>📊 Model Info</h2>
            <div class="endpoint">
                <strong>Mode:</strong> {'MOCK' if use_mock else 'PRODUCTION'}<br>
                <strong>Classes:</strong> {len(classifier.labels) if classifier else 0}<br>
                <strong>Model:</strong> {'Loaded' if not use_mock else 'Not loaded (using mock)'}
            </div>
            
            <p style="margin-top: 30px; font-size: 0.9em;">
                Created by Roy Kiprop Kongwalei | Brandeis University
            </p>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html)

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "mode": "mock" if use_mock else "production",
        "model_loaded": not use_mock,
        "classes_available": len(classifier.labels) if classifier else 0
    }

@app.get("/classes")
async def get_classes():
    """Get all pest classes"""
    if not classifier:
        raise HTTPException(status_code=503, detail="Classifier not initialized")
    
    return {
        "classes": classifier.labels,
        "count": len(classifier.labels),
        "mode": "mock" if use_mock else "production"
    }

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    include_treatment: bool = True,
    background_tasks: BackgroundTasks = None
):
    """
    Predict pest from uploaded image
    
    - **file**: Image file (JPG, PNG, etc.)
    - **include_treatment**: Include treatment recommendations
    """
    start_time = time.time()
    
    try:
        # Validate file
        is_valid, error_msg = validate_image(file)
        if not is_valid:
            raise HTTPException(status_code=400, detail=error_msg)
        
        # Check file size
        contents = await file.read()
        if len(contents) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400, 
                detail=f"File too large. Max size: {MAX_FILE_SIZE / (1024*1024)}MB"
            )
        
        # Open image
        try:
            image = Image.open(io.BytesIO(contents))
            # Convert RGBA to RGB if necessary
            if image.mode in ('RGBA', 'LA', 'P'):
                image = image.convert('RGB')
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")
        
        # Get predictions
        if not classifier:
            raise HTTPException(status_code=503, detail="Classifier not initialized")
        
        predictions = classifier.predict(image)
        
        # Add treatment info if requested
        if include_treatment:
            for pred in predictions:
                pred['treatment'] = get_treatment_info(pred['class'])
        
        # Save uploaded file asynchronously (optional)
        if background_tasks:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{timestamp}_{file.filename}"
            filepath = os.path.join(UPLOAD_DIR, filename)
            background_tasks.add_task(save_uploaded_file, contents, filepath)
        
        processing_time = time.time() - start_time
        
        return JSONResponse({
            "success": True,
            "filename": file.filename,
            "file_size": len(contents),
            "predictions": predictions,
            "processing_time": round(processing_time, 3),
            "mode": "mock" if use_mock else "production",
            "timestamp": datetime.now().isoformat()
        })
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict/batch")
async def predict_batch(
    files: List[UploadFile] = File(...),
    include_treatment: bool = True
):
    """
    Process multiple images at once
    
    - **files**: List of image files
    - **include_treatment**: Include treatment recommendations
    """
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 files per batch")
    
    results = []
    failed = 0
    
    for file in files:
        try:
            # Validate file
            is_valid, error_msg = validate_image(file)
            if not is_valid:
                results.append({
                    "filename": file.filename,
                    "success": False,
                    "error": error_msg
                })
                failed += 1
                continue
            
            # Read and process
            contents = await file.read()
            if len(contents) > MAX_FILE_SIZE:
                results.append({
                    "filename": file.filename,
                    "success": False,
                    "error": f"File too large. Max size: {MAX_FILE_SIZE / (1024*1024)}MB"
                })
                failed += 1
                continue
            
            image = Image.open(io.BytesIO(contents))
            if image.mode in ('RGBA', 'LA', 'P'):
                image = image.convert('RGB')
            
            # Get predictions
            predictions = classifier.predict(image)
            
            if include_treatment:
                for pred in predictions:
                    pred['treatment'] = get_treatment_info(pred['class'])
            
            results.append({
                "filename": file.filename,
                "success": True,
                "predictions": predictions
            })
            
        except Exception as e:
            results.append({
                "filename": file.filename,
                "success": False,
                "error": str(e)
            })
            failed += 1
    
    return {
        "results": results,
        "total": len(files),
        "successful": len(files) - failed,
        "failed": failed,
        "mode": "mock" if use_mock else "production"
    }

@app.get("/stats")
async def get_stats():
    """Get API statistics"""
    return {
        "api_version": "2.0.0",
        "mode": "mock" if use_mock else "production",
        "classes_count": len(classifier.labels) if classifier else 0,
        "model_path": MODEL_PATH if os.path.exists(MODEL_PATH) else None,
        "labels_path": LABELS_PATH if os.path.exists(LABELS_PATH) else None,
        "upload_dir": UPLOAD_DIR,
        "max_file_size_mb": MAX_FILE_SIZE / (1024 * 1024)
    }

# Helper function for async file saving
async def save_uploaded_file(contents: bytes, filepath: str):
    """Save uploaded file asynchronously"""
    try:
        with open(filepath, 'wb') as f:
            f.write(contents)
        logger.info(f"Saved uploaded file: {filepath}")
    except Exception as e:
        logger.error(f"Error saving file {filepath}: {e}")

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": exc.detail,
            "timestamp": datetime.now().isoformat()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Internal server error",
            "timestamp": datetime.now().isoformat()
        }
    )

if __name__ == "__main__":
    import numpy as np  # Import here to avoid circular imports
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    
    logger.info(f"Starting server on {host}:{port}")
    uvicorn.run(
        "app:app",
        host=host,
        port=port,
        reload=True,
        log_level="info"
    )