"""
Lightweight Deepfake Detection - No PyTorch Required
Installs in seconds, uses only OpenCV and basic CV techniques
"""

import os
import tempfile
import logging
from typing import Dict, Any
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DetectionResponse(BaseModel):
    prediction: str
    confidence: float
    probability: float
    explanation: str
    details: Dict[str, Any]

class SimpleDeepfakeDetector:
    """Lightweight detector using only OpenCV"""
    
    def __init__(self):
        logger.info("Initializing Simple Deepfake Detector...")
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        logger.info("✅ Simple Detector ready!")
    
    def analyze_image_quality(self, image_array):
        """Analyze image quality metrics"""
        scores = {}
        
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        
        # Sharpness (Laplacian variance)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        scores['sharpness'] = min(100, laplacian_var / 10)
        
        # Compression artifacts (edge density)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = (np.sum(edges > 0) / edges.size) * 100
        scores['compression_artifacts'] = edge_density
        
        # Color consistency
        hsv = cv2.cvtColor(image_array, cv2.COLOR_RGB2HSV)
        color_variance = np.std(hsv[:,:,0])
        scores['color_consistency'] = min(100, 100 - (color_variance / 2))
        
        # Lighting consistency
        brightness = hsv[:,:,2]
        brightness_std = np.std(brightness)
        scores['lighting_consistency'] = min(100, 100 - brightness_std / 3)
        
        # Face detection
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        scores['face_detected'] = len(faces) > 0
        scores['num_faces'] = len(faces)
        
        return scores
    
    def calculate_probability(self, quality_scores):
        """Calculate deepfake probability"""
        anomaly_score = 0.0
        
        if quality_scores['sharpness'] < 50:
            anomaly_score += 0.15 * (50 - quality_scores['sharpness']) / 50
        
        if quality_scores['compression_artifacts'] > 30:
            anomaly_score += 0.25 * (quality_scores['compression_artifacts'] - 30) / 70
        
        if quality_scores['color_consistency'] < 60:
            anomaly_score += 0.20 * (60 - quality_scores['color_consistency']) / 60
        
        if quality_scores['lighting_consistency'] < 60:
            anomaly_score += 0.25 * (60 - quality_scores['lighting_consistency']) / 60
        
        if not quality_scores['face_detected']:
            anomaly_score += 0.075
        elif quality_scores['num_faces'] > 1:
            anomaly_score += 0.045
        
        return min(1.0, anomaly_score)
    
    def analyze(self, image_path):
        """Analyze image"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                from PIL import Image
                pil_image = Image.open(image_path).convert('RGB')
                image = np.array(pil_image)
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            quality_scores = self.analyze_image_quality(image)
            probability = self.calculate_probability(quality_scores)
            
            if probability < 0.3:
                prediction = "likely authentic"
                explanation = "Image shows consistent quality metrics typical of authentic media"
            elif probability < 0.6:
                prediction = "possibly manipulated"
                explanation = "Some quality inconsistencies detected that may indicate manipulation"
            else:
                prediction = "likely deepfake"
                explanation = "Multiple anomalies detected suggesting AI-generated or manipulated content"
            
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

app = FastAPI(title="Simple Deepfake Detection", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

detector = None

@app.on_event("startup")
async def startup():
    global detector
    detector = SimpleDeepfakeDetector()

@app.post("/analyze", response_model=DetectionResponse)
async def analyze(file: UploadFile = File(...)):
    allowed = ['image/jpeg', 'image/jpg', 'image/png', 'image/webp']
    if file.content_type not in allowed:
        raise HTTPException(400, "Unsupported file type")
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp:
        content = await file.read()
        temp.write(content)
        temp_path = temp.name
    
    try:
        result = detector.analyze(temp_path)
        return DetectionResponse(**result)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.get("/health")
async def health():
    return {"status": "healthy", "model": "OpenCV-only (Lightweight)"}

if __name__ == "__main__":
    print("🚀 Starting Simple Deepfake API on http://localhost:5000")
    uvicorn.run(app, host="0.0.0.0", port=5000)

