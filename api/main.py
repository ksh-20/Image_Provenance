"""
Unified API that serves both:
  1. Image Provenance Analysis (graph construction + manipulation tracing)
  2. Deepfake Detection (multi-modal: visual + audio + text)
  3. Combined analysis: provenance-conditioned deepfake scoring
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import torch
import numpy as np
import cv2
import tempfile
import os
import uuid
import base64
import json
import io
from PIL import Image

# ── Local imports ──
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.provenance_model import ProvenanceAnalysisModel, kruskal_mst
from models.deepfake_detector import (
    GuardianEyeDeepfakeDetector,
    ProvenanceConditionedScoring,
    VisualFeatureExtractor
)

app = FastAPI(
    title="Guardian Eye + Provenance Analysis API",
    description="""
    Unified API for:
    - Image Provenance Analysis via Graph Encoding with Vision Transformer
    - Multi-Modal Deepfake Detection (Guardian Eye)
    - Combined provenance-conditioned deepfake scoring
    """,
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global model instances ──
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
provenance_model: Optional[ProvenanceAnalysisModel] = None
deepfake_detector: Optional[GuardianEyeDeepfakeDetector] = None
provenance_scorer = ProvenanceConditionedScoring()


def get_provenance_model() -> ProvenanceAnalysisModel:
    global provenance_model
    if provenance_model is None:
        provenance_model = ProvenanceAnalysisModel(
            lora_rank=16,
            hidden_dim=768,
            num_graph_layers=3,
            num_clusters=8
        ).to(DEVICE)
        provenance_model.eval()
        # Load weights if available
        weights_path = os.environ.get('PROVENANCE_WEIGHTS', 'weights/provenance_best.pth')
        if os.path.exists(weights_path):
            state = torch.load(weights_path, map_location=DEVICE)
            provenance_model.load_state_dict(state, strict=False)
    return provenance_model


def get_deepfake_detector() -> GuardianEyeDeepfakeDetector:
    global deepfake_detector
    if deepfake_detector is None:
        model_path = os.environ.get('DEEPFAKE_MODEL', 'weights/guardian_eye_rf.pkl')
        scaler_path = os.environ.get('DEEPFAKE_SCALER', 'weights/guardian_eye_scaler.pkl')
        deepfake_detector = GuardianEyeDeepfakeDetector(
            model_path=model_path if os.path.exists(model_path) else None,
            scaler_path=scaler_path if os.path.exists(scaler_path) else None,
            device=DEVICE
        )
    return deepfake_detector


# ─────────────────────────────────────────────
# Pydantic Models
# ─────────────────────────────────────────────

class ProvenanceResult(BaseModel):
    case_id: str
    num_images: int
    undirected_graph: List[List[float]]
    directed_graph: List[List[float]]
    edge_probabilities: List[List[float]]
    edge_uncertainties: List[List[float]]
    direction_matrix: List[List[float]]
    cluster_assignments: List[List[float]]
    frequency_scores: List[float]
    manipulation_suspected: List[bool]
    root_images: List[int]
    leaf_images: List[int]


class DeepfakeResult(BaseModel):
    file_id: str
    prediction: str
    probability_fake: float
    probability_real: float
    confidence: float
    modality_importance: Optional[Dict[str, float]]


class CombinedResult(BaseModel):
    case_id: str
    provenance: ProvenanceResult
    deepfake_scores: List[DeepfakeResult]
    provenance_adjusted_scores: List[float]
    high_risk_images: List[int]
    analysis_summary: str


class HealthResponse(BaseModel):
    status: str
    device: str
    models_loaded: Dict[str, bool]


# ─────────────────────────────────────────────
# Utility Functions
# ─────────────────────────────────────────────

def decode_image_b64(b64_string: str) -> np.ndarray:
    """Decode base64 image to numpy array (BGR)."""
    img_data = base64.b64decode(b64_string)
    nparr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img


def preprocess_image_for_model(image: np.ndarray) -> torch.Tensor:
    """
    image: (H, W, 3) BGR numpy
    Returns: (1, 3, 224, 224) tensor
    """
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (224, 224))
    img_float = img_resized.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_norm = (img_float - mean) / std
    tensor = torch.from_numpy(img_norm.transpose(2, 0, 1)).float().unsqueeze(0)
    return tensor.to(DEVICE)


def identify_roots_leaves(adj_directed: np.ndarray):
    """Identify root (no parents) and leaf (no children) nodes."""
    n = adj_directed.shape[0]
    in_degree = adj_directed.sum(axis=0)
    out_degree = adj_directed.sum(axis=1)
    roots = [int(i) for i in range(n) if in_degree[i] == 0]
    leaves = [int(i) for i in range(n) if out_degree[i] == 0]
    return roots, leaves


def generate_summary(roots, leaves, high_risk, num_images, prov_scores):
    """Generate a human-readable analysis summary."""
    lines = [
        f"Provenance graph contains {num_images} related images.",
        f"Found {len(roots)} root image(s) (idx: {roots}) and {len(leaves)} leaf image(s) (idx: {leaves}).",
    ]
    if high_risk:
        lines.append(
            f"⚠️  {len(high_risk)} image(s) flagged as high-risk for deepfake manipulation "
            f"(idx: {high_risk})."
        )
        avg_score = np.mean([prov_scores[i] for i in high_risk])
        lines.append(f"   Average manipulation score for high-risk images: {avg_score:.2%}")
    else:
        lines.append("✅ No high-risk deepfake manipulation detected.")
    lines.append(
        "Analysis uses ViT-LoRA graph encoding + MFCC/BERT multi-modal fusion + "
        "provenance-conditioned scoring."
    )
    return " ".join(lines)


# ─────────────────────────────────────────────
# API Endpoints
# ─────────────────────────────────────────────

@app.get("/", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="running",
        device=DEVICE,
        models_loaded={
            "provenance": provenance_model is not None,
            "deepfake": deepfake_detector is not None
        }
    )


@app.post("/analyze/provenance", response_model=ProvenanceResult)
async def analyze_provenance(
    images: List[UploadFile] = File(..., description="Upload 3-50 related images")
):
    """
    Run provenance graph analysis on a set of semantically-similar images.
    Returns directed provenance graph with manipulation history.
    """
    if len(images) < 2:
        raise HTTPException(status_code=400, detail="At least 2 images required.")
    if len(images) > 50:
        raise HTTPException(status_code=400, detail="Maximum 50 images per request.")

    case_id = str(uuid.uuid4())
    model = get_provenance_model()
    image_tensors = []
    raw_images = []

    for upload in images:
        content = await upload.read()
        nparr = np.frombuffer(content, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(status_code=400,
                                detail=f"Could not decode image: {upload.filename}")
        raw_images.append(img)
        tensor = preprocess_image_for_model(img)
        image_tensors.append(tensor)

    # Stack all images
    image_batch = torch.cat(image_tensors, dim=0)  # (N, 3, 224, 224)

    with torch.no_grad():
        output = model(image_batch, mode='inference')

    adj_undirected = output['adj_undirected'].cpu().numpy()
    adj_directed = output['adj_directed'].cpu().numpy()
    dir_matrix = output['direction_matrix'].cpu().numpy()
    edge_probs = output['edge_probabilities'].squeeze(0).cpu().numpy()
    edge_vars = output['edge_uncertainties'].squeeze(0).cpu().numpy()
    freq_scores = output['freq_scores'].cpu().numpy().tolist()
    cluster_assign = output['cluster_assignments'].squeeze(0).cpu().numpy()

    roots, leaves = identify_roots_leaves(adj_directed)
    manipulation_suspected = [bool(s > 0.6) for s in freq_scores]

    return ProvenanceResult(
        case_id=case_id,
        num_images=len(images),
        undirected_graph=adj_undirected.tolist(),
        directed_graph=adj_directed.tolist(),
        edge_probabilities=edge_probs.tolist(),
        edge_uncertainties=edge_vars.tolist(),
        direction_matrix=dir_matrix.tolist(),
        cluster_assignments=cluster_assign.tolist(),
        frequency_scores=freq_scores,
        manipulation_suspected=manipulation_suspected,
        root_images=roots,
        leaf_images=leaves
    )


@app.post("/analyze/deepfake/video", response_model=DeepfakeResult)
async def analyze_deepfake_video(video: UploadFile = File(...)):
    """
    Run Guardian Eye multi-modal deepfake detection on a video file.
    """
    detector = get_deepfake_detector()
    # Save to temp file
    suffix = os.path.splitext(video.filename)[-1] or '.mp4'
    fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    try:
        content = await video.read()
        with os.fdopen(fd, 'wb') as f:
            f.write(content)
        result = detector.predict(tmp_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    return DeepfakeResult(
        file_id=str(uuid.uuid4()),
        prediction=result['prediction'],
        probability_fake=result.get('probability_fake', 0.5),
        probability_real=result.get('probability_real', 0.5),
        confidence=result.get('confidence', 0.0),
        modality_importance=result.get('modality_importance')
    )


@app.post("/analyze/deepfake/image", response_model=DeepfakeResult)
async def analyze_deepfake_image(image: UploadFile = File(...)):
    """
    Run deepfake detection on a single image.
    """
    detector = get_deepfake_detector()
    content = await image.read()
    nparr = np.frombuffer(content, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode image.")
    result = detector.predict_image(img)
    return DeepfakeResult(
        file_id=str(uuid.uuid4()),
        prediction=result['prediction'],
        probability_fake=result.get('probability_fake', 0.5),
        probability_real=1.0 - result.get('probability_fake', 0.5),
        confidence=result.get('confidence', 0.0),
        modality_importance=None
    )


@app.post("/analyze/combined", response_model=CombinedResult)
async def analyze_combined(
    images: List[UploadFile] = File(...,
        description="Upload related images for combined provenance + deepfake analysis")
):
    """
    Combined pipeline:
      1. Build provenance graph
      2. Run deepfake detection on each image
      3. Adjust deepfake scores by provenance topology (NOVELTY #4)
      4. Return unified risk assessment
    """
    if len(images) < 2:
        raise HTTPException(status_code=400, detail="At least 2 images required.")

    case_id = str(uuid.uuid4())
    model = get_provenance_model()
    detector = get_deepfake_detector()

    image_tensors = []
    raw_images = []
    for upload in images:
        content = await upload.read()
        nparr = np.frombuffer(content, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(400, f"Cannot decode: {upload.filename}")
        raw_images.append(img)
        image_tensors.append(preprocess_image_for_model(img))

    image_batch = torch.cat(image_tensors, dim=0)

    # ── Provenance analysis ──
    with torch.no_grad():
        prov_output = model(image_batch, mode='inference')

    adj_undirected = prov_output['adj_undirected'].cpu().numpy()
    adj_directed = prov_output['adj_directed'].cpu().numpy()
    freq_scores = prov_output['freq_scores'].cpu().numpy().tolist()
    edge_probs = prov_output['edge_probabilities'].squeeze(0).cpu().numpy()
    edge_vars = prov_output['edge_uncertainties'].squeeze(0).cpu().numpy()
    dir_matrix = prov_output['direction_matrix'].cpu().numpy()
    cluster_assign = prov_output['cluster_assignments'].squeeze(0).cpu().numpy()

    roots, leaves = identify_roots_leaves(adj_directed)

    prov_result = ProvenanceResult(
        case_id=case_id,
        num_images=len(images),
        undirected_graph=adj_undirected.tolist(),
        directed_graph=adj_directed.tolist(),
        edge_probabilities=edge_probs.tolist(),
        edge_uncertainties=edge_vars.tolist(),
        direction_matrix=dir_matrix.tolist(),
        cluster_assignments=cluster_assign.tolist(),
        frequency_scores=freq_scores,
        manipulation_suspected=[bool(s > 0.6) for s in freq_scores],
        root_images=roots,
        leaf_images=leaves
    )

    # ── Deepfake detection per image ──
    deepfake_results = []
    raw_df_scores = []
    for img in raw_images:
        r = detector.predict_image(img)
        deepfake_results.append(DeepfakeResult(
            file_id=str(uuid.uuid4()),
            prediction=r['prediction'],
            probability_fake=r.get('probability_fake', 0.5),
            probability_real=1.0 - r.get('probability_fake', 0.5),
            confidence=r.get('confidence', 0.0),
            modality_importance=None
        ))
        raw_df_scores.append(r.get('probability_fake', 0.5))

    # ── Provenance-conditioned scoring (NOVELTY #4) ──
    raw_scores_np = np.array(raw_df_scores)
    adjusted_scores = provenance_scorer.apply(raw_scores_np, adj_directed)
    high_risk = [int(i) for i, s in enumerate(adjusted_scores) if s > 0.65]

    summary = generate_summary(roots, leaves, high_risk,
                               len(images), adjusted_scores)

    return CombinedResult(
        case_id=case_id,
        provenance=prov_result,
        deepfake_scores=deepfake_results,
        provenance_adjusted_scores=adjusted_scores.tolist(),
        high_risk_images=high_risk,
        analysis_summary=summary
    )


@app.get("/model/info")
async def model_info():
    """Return model architecture details."""
    return {
        "provenance_model": {
            "backbone": "ViT-B/16 with LoRA (rank=16)",
            "graph_encoder": "Graph Structure Masked Attention Transformer (3 layers)",
            "link_prediction": "Weighted Patch Distance + Kruskal MST",
            "direction": "Precedence embeddings with virtual source/target nodes",
            "novelties": [
                "Frequency-domain cross-modal consistency (DCT residuals)",
                "Adaptive edge confidence with MC dropout uncertainty",
                "Hierarchical cluster-then-reason (k=8)",
                "Provenance-conditioned deepfake scoring"
            ]
        },
        "deepfake_model": {
            "visual": "ResNet-18 → 512-dim",
            "audio": "MFCC (13 coeff)",
            "text": "BERT-base → 768-dim",
            "fusion_dim": 1293,
            "classifier": "Random Forest (100 trees)"
        },
        "device": DEVICE
    }