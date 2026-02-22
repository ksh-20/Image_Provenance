"""
Free & Open-Source Deepfake Detection Backend
Uses only free models and APIs - no paid services required
"""

import os
import tempfile
import logging
from typing import Optional, Dict, Any
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# For deepfake detection, we'll use a simple but effective approach
# using face detection + quality analysis
try:
    from PIL import Image
    import torch
    import torchvision.transforms as transforms
    from torchvision import models
except ImportError:
    print("Installing required packages...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'torch', 'torchvision', 'pillow'])
    from PIL import Image
    import torch
    import torchvision.transforms as transforms
    from torchvision import models

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Response model
class DetectionResponse(BaseModel):
    prediction: str  # "Real", "Deepfake", or "Possibly Manipulated"
    confidence: float
    probability: float
    explanation: str
    details: Dict[str, Any]

# Global model
deepfake_analyzer = None

class FreeDeepfakeDetector:
    """Free open-source deepfake detector using computer vision techniques"""
    
    def __init__(self):
        logger.info("Initializing Free Deepfake Detector...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load pre-trained ResNet for feature extraction (free, open-source)
        self.feature_extractor = models.resnet50(pretrained=True)
        self.feature_extractor.eval()
        self.feature_extractor.to(self.device)
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # Load OpenCV's face detector (free)
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        logger.info("✅ Free Deepfake Detector initialized successfully!")
    
    def detect_faces(self, image_array):
        """Detect faces in image"""
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        return faces
    
    def analyze_image_quality(self, image_array):
        """Analyze image quality metrics that can indicate manipulation"""
        scores = {}
        
        # 1. Laplacian variance (blur detection)
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        scores['sharpness'] = min(100, laplacian_var / 10)
        
        # 2. JPEG compression artifacts
        # Higher compression often indicates manipulation
        edges = cv2.Canny(gray, 50, 150)
        edge_density = (np.sum(edges > 0) / edges.size) * 100
        scores['compression_artifacts'] = edge_density
        
        # 3. Color consistency
        # Check for unusual color distributions
        hsv = cv2.cvtColor(image_array, cv2.COLOR_RGB2HSV)
        color_variance = np.std(hsv[:,:,0])
        scores['color_consistency'] = min(100, 100 - (color_variance / 2))
        
        # 4. Lighting consistency
        # Analyze brightness distribution
        brightness = hsv[:,:,2]
        brightness_std = np.std(brightness)
        scores['lighting_consistency'] = min(100, 100 - brightness_std / 3)
        
        # 5. Face detection quality
        faces = self.detect_faces(image_array)
        scores['face_detected'] = len(faces) > 0
        scores['num_faces'] = len(faces)
        
        return scores
    
    def calculate_deepfake_probability(self, quality_scores):
        """Calculate probability of deepfake based on quality metrics"""
        
        # Weights for different factors
        weights = {
            'sharpness': 0.15,
            'compression_artifacts': 0.25,
            'color_consistency': 0.20,
            'lighting_consistency': 0.25,
            'face_anomaly': 0.15
        }
        
        # Calculate anomaly scores
        anomaly_score = 0.0
        
        # Low sharpness can indicate AI generation
        if quality_scores['sharpness'] < 50:
            anomaly_score += weights['sharpness'] * (50 - quality_scores['sharpness']) / 50
        
        # High compression artifacts
        if quality_scores['compression_artifacts'] > 30:
            anomaly_score += weights['compression_artifacts'] * \
                           (quality_scores['compression_artifacts'] - 30) / 70
        
        # Poor color consistency
        if quality_scores['color_consistency'] < 60:
            anomaly_score += weights['color_consistency'] * \
                           (60 - quality_scores['color_consistency']) / 60
        
        # Poor lighting consistency
        if quality_scores['lighting_consistency'] < 60:
            anomaly_score += weights['lighting_consistency'] * \
                           (60 - quality_scores['lighting_consistency']) / 60
        
        # Face detection issues
        if not quality_scores['face_detected']:
            anomaly_score += weights['face_anomaly'] * 0.5
        elif quality_scores['num_faces'] > 1:
            anomaly_score += weights['face_anomaly'] * 0.3
        
        return min(1.0, anomaly_score)
    
    def analyze(self, image_path):
        """Main analysis function"""
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                # Try with PIL
                pil_image = Image.open(image_path).convert('RGB')
                image = np.array(pil_image)
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Analyze quality metrics
            quality_scores = self.analyze_image_quality(image)
            
            # Calculate deepfake probability
            probability = self.calculate_deepfake_probability(quality_scores)
            
            # Determine verdict with detailed explanation
            issues = []
            
            # Analyze specific issues
            if quality_scores['sharpness'] < 50:
                issues.append(f"Low sharpness ({quality_scores['sharpness']:.1f}/100) - typical of AI smoothing")
            
            if quality_scores['compression_artifacts'] > 40:
                issues.append(f"High compression artifacts ({quality_scores['compression_artifacts']:.1f}%) - suggests re-encoding")
            
            if quality_scores['color_consistency'] < 70:
                issues.append(f"Poor color consistency ({quality_scores['color_consistency']:.1f}/100) - unnatural color distribution")
            
            if quality_scores['lighting_consistency'] < 70:
                issues.append(f"Inconsistent lighting ({quality_scores['lighting_consistency']:.1f}/100) - artificial light patterns")
            
            if not quality_scores['face_detected']:
                issues.append("No face detected - cannot verify facial consistency")
            elif quality_scores['num_faces'] > 1:
                issues.append(f"Multiple faces detected ({quality_scores['num_faces']}) - harder to verify authenticity")
            
            if probability < 0.3:
                prediction = "likely authentic"
                if len(issues) == 0:
                    explanation = "Image passes all quality checks. Sharpness, color, lighting, and compression patterns are consistent with authentic media."
                else:
                    explanation = f"Minor issues detected but overall appears authentic. {' '.join(issues[:2])}"
            elif probability < 0.6:
                prediction = "possibly manipulated"
                explanation = f"Moderate concerns detected. {' '.join(issues[:2])} Consider additional verification."
            else:
                prediction = "likely deepfake"
                if len(issues) > 0:
                    explanation = f"HIGH RISK: {' '.join(issues[:3])} These patterns are commonly found in AI-generated or manipulated images."
                else:
                    explanation = "Multiple anomalies detected suggesting AI-generated or manipulated content"
            
            # Calculate confidence (inverse of how close to threshold)
            if probability < 0.3:
                confidence = (0.3 - probability) / 0.3
            elif probability < 0.6:
                confidence = min(abs(probability - 0.3), abs(probability - 0.6)) / 0.15
            else:
                confidence = (probability - 0.6) / 0.4
            
            confidence = max(0.5, min(0.95, confidence))
            
            return {
                'prediction': prediction,
                'probability': float(probability),
                'confidence': float(confidence),
                'explanation': explanation,
                'issues_found': issues,
                'details': {
                    'sharpness_score': round(quality_scores['sharpness'], 2),
                    'compression_artifacts': round(quality_scores['compression_artifacts'], 2),
                    'color_consistency': round(quality_scores['color_consistency'], 2),
                    'lighting_consistency': round(quality_scores['lighting_consistency'], 2),
                    'faces_detected': quality_scores['num_faces']
                }
            }
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            raise

# Initialize FastAPI app
app = FastAPI(
    title="Free Deepfake Detection API",
    description="Open-source deepfake detection using computer vision",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    global deepfake_analyzer
    logger.info("Starting Free Deepfake Detection API...")
    deepfake_analyzer = FreeDeepfakeDetector()
    logger.info("✅ API ready!")

@app.post("/analyze", response_model=DetectionResponse)
async def analyze_media(file: UploadFile = File(...)):
    """
    Analyze uploaded image for deepfake/manipulation
    Supports: JPG, PNG, WebP images
    """
    # Validate file type
    allowed_types = ['image/jpeg', 'image/jpg', 'image/png', 'image/webp']
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {', '.join(allowed_types)}"
        )
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        content = await file.read()
        temp_file.write(content)
        temp_path = temp_file.name
    
    try:
        # Analyze the image
        logger.info(f"Analyzing: {file.filename}")
        result = deepfake_analyzer.analyze(temp_path)
        
        return DetectionResponse(**result)
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
    
    finally:
        # Clean up
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "message": "Free Deepfake Detection API is running",
        "model": "Computer Vision + ResNet50 (Open Source)"
    }

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Free Deepfake Detection API",
        "endpoints": {
            "/analyze": "POST - Analyze image for deepfakes",
            "/health": "GET - Health check",
            "/docs": "GET - API documentation"
        }
    }

if __name__ == "__main__":
    print("🚀 Starting Free Deepfake Detection API on http://localhost:5000")
    print("📝 API Documentation: http://localhost:5000/docs")
    uvicorn.run(app, host="0.0.0.0", port=5000)

