"""
Integrates with the Provenance Analysis system to provide a dual-pipeline:
  - Provenance Pipeline: traces editing history via graph analysis
  - Deepfake Pipeline:   classifies real vs. fake using multi-modal fusion

Architecture (from Guardian Eye paper):
  Visual:  ResNet-18 → 512-dim embeddings
  Audio:   MFCC (13 coeff) → 13-dim features
  Text:    BERT → 768-dim embeddings
  Fusion:  Concat → 1293-dim → Random Forest

Novelty additions in this module:
  - Provenance-Conditioned Deepfake Scoring: uses provenance graph topology
    to weight deepfake detection scores (nodes closer to root = more trusted)
  - Ensemble uncertainty from MC dropout passed to API
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models
import numpy as np
from transformers import BertModel, BertTokenizer, Wav2Vec2ForCTC, Wav2Vec2Processor
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import librosa
import cv2
import tempfile
import os
import joblib
from typing import Optional, Dict, Any, List, Tuple


# ─────────────────────────────────────────────
# Visual Feature Extractor (ResNet-18)
# ─────────────────────────────────────────────

class VisualFeatureExtractor(nn.Module):
    """ResNet-18 without final FC — outputs 512-dim frame embeddings."""

    def __init__(self):
        super().__init__()
        resnet = tv_models.resnet18(weights=tv_models.ResNet18_Weights.IMAGENET1K_V1)
        # Remove final classification layer
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """
        frames: (B, T, 3, 224, 224) or (B, 3, 224, 224)
        Returns: (B, 512) aggregated visual embedding
        """
        if frames.dim() == 5:
            B, T, C, H, W = frames.shape
            x = frames.view(B * T, C, H, W)
            feats = self.backbone(x)           # (B*T, 512, 1, 1)
            feats = feats.view(B, T, 512)
            return feats.mean(dim=1)           # temporal average pooling
        else:
            feats = self.backbone(frames)      # (B, 512, 1, 1)
            return feats.view(frames.shape[0], -1)  # (B, 512)


# ─────────────────────────────────────────────
# Audio Feature Extractor (MFCC)
# ─────────────────────────────────────────────

class AudioFeatureExtractor:
    """Extracts 13-dimensional MFCC features from audio waveforms."""

    def __init__(self, n_mfcc: int = 13, sample_rate: int = 44100):
        self.n_mfcc = n_mfcc
        self.sample_rate = sample_rate

    def extract(self, audio_path: str) -> np.ndarray:
        """
        audio_path: path to WAV file
        Returns: (13,) average MFCC vector
        """
        try:
            y, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc)
            return mfcc.mean(axis=1)   # (13,)
        except Exception:
            return np.zeros(self.n_mfcc)

    def extract_from_array(self, audio_array: np.ndarray,
                           sr: int = 44100) -> np.ndarray:
        mfcc = librosa.feature.mfcc(y=audio_array.astype(np.float32),
                                    sr=sr, n_mfcc=self.n_mfcc)
        return mfcc.mean(axis=1)


# ─────────────────────────────────────────────
# Text Feature Extractor (BERT)
# ─────────────────────────────────────────────

class TextFeatureExtractor(nn.Module):
    """BERT-based text embeddings from speech transcription."""

    def __init__(self, model_name: str = 'bert-base-uncased', max_length: int = 512):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.bert = BertModel.from_pretrained(model_name)
        self.max_length = max_length
        # Freeze BERT for inference efficiency
        for p in self.bert.parameters():
            p.requires_grad_(False)

    def forward(self, text: str) -> torch.Tensor:
        """
        text: transcribed speech string
        Returns: (768,) sentence embedding
        """
        if not text or text.strip() == "":
            return torch.zeros(768)
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            max_length=self.max_length,
            truncation=True,
            padding='max_length'
        )
        with torch.no_grad():
            outputs = self.bert(**inputs)
        # Average last hidden states → sentence vector
        hidden = outputs.last_hidden_state  # (1, seq_len, 768)
        return hidden.mean(dim=1).squeeze(0)  # (768,)


# ─────────────────────────────────────────────
# Speech Recognition (Wav2Vec2)
# ─────────────────────────────────────────────

class SpeechTranscriber:
    """Transcribes audio to text using Wav2Vec2."""

    def __init__(self, model_name: str = 'facebook/wav2vec2-base-960h'):
        self.model_name = model_name
        self._processor = None
        self._model = None

    def _load(self):
        if self._processor is None:
            self._processor = Wav2Vec2Processor.from_pretrained(self.model_name)
            self._model = Wav2Vec2ForCTC.from_pretrained(self.model_name)
            self._model.eval()

    def transcribe(self, audio_path: str, sample_rate: int = 16000) -> str:
        """Returns transcription string or empty string on failure."""
        try:
            self._load()
            import soundfile as sf
            speech, sr = sf.read(audio_path)
            if sr != sample_rate:
                import resampy
                speech = resampy.resample(speech, sr, sample_rate)
            inputs = self._processor(
                speech, sampling_rate=sample_rate, return_tensors='pt', padding=True
            )
            with torch.no_grad():
                logits = self._model(inputs.input_values).logits
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = self._processor.batch_decode(predicted_ids)[0]
            return transcription.lower()
        except Exception:
            return ""


# ─────────────────────────────────────────────
# Multi-Modal Fusion + Random Forest Classifier
# ─────────────────────────────────────────────

class GuardianEyeDeepfakeDetector:
    """
    Guardian Eye multi-modal deepfake detector.
    Integrates visual (ResNet-18) + audio (MFCC) + text (BERT) features
    fused into 1293-dim vector, classified by Random Forest.

    Feature dimension: 512 (visual) + 13 (audio) + 768 (text) = 1293
    """

    VISUAL_DIM = 512
    AUDIO_DIM = 13
    TEXT_DIM = 768
    FEATURE_DIM = VISUAL_DIM + AUDIO_DIM + TEXT_DIM  # 1293

    def __init__(self,
                 model_path: Optional[str] = None,
                 scaler_path: Optional[str] = None,
                 device: str = 'cpu'):
        self.device = device
        # Sub-extractors
        self.visual_extractor = VisualFeatureExtractor().to(device)
        self.visual_extractor.eval()
        self.audio_extractor = AudioFeatureExtractor()
        self.text_extractor = TextFeatureExtractor().to(device)
        self.transcriber = SpeechTranscriber()
        # Classifier
        self.rf_model: Optional[RandomForestClassifier] = None
        self.scaler = StandardScaler()
        if model_path and os.path.exists(model_path):
            self.rf_model = joblib.load(model_path)
        if scaler_path and os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)

    # ── Video processing ──

    def extract_frames(self, video_path: str,
                       fps: int = 1, max_frames: int = 30) -> np.ndarray:
        """Extract frames at 1 fps, max 30 frames."""
        cap = cv2.VideoCapture(video_path)
        video_fps = cap.get(cv2.CAP_PROP_FPS) or 25
        interval = max(1, int(video_fps / fps))
        frames = []
        frame_idx = 0
        while len(frames) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % interval == 0:
                frame = cv2.resize(frame, (224, 224))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            frame_idx += 1
        cap.release()
        if not frames:
            return np.zeros((1, 224, 224, 3), dtype=np.uint8)
        return np.stack(frames)  # (T, H, W, 3)

    def preprocess_frames(self, frames: np.ndarray) -> torch.Tensor:
        """Normalize and convert frames to tensor."""
        frames = frames.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        frames = (frames - mean) / std
        frames = np.transpose(frames, (0, 3, 1, 2))  # (T, 3, H, W)
        return torch.from_numpy(frames).float().to(self.device)

    def extract_audio(self, video_path: str,
                      output_path: Optional[str] = None) -> Optional[str]:
        """Extract audio from video using ffmpeg."""
        if output_path is None:
            fd, output_path = tempfile.mkstemp(suffix='.wav')
            os.close(fd)
        cmd = (f'ffmpeg -i "{video_path}" -ar 44100 -ac 1 -acodec pcm_s16le '
               f'"{output_path}" -y -loglevel quiet')
        ret = os.system(cmd)
        if ret == 0 and os.path.exists(output_path):
            return output_path
        return None

    # ── Feature extraction ──

    def extract_visual_features(self, video_path: str) -> np.ndarray:
        frames = self.extract_frames(video_path)
        frame_tensor = self.preprocess_frames(frames)  # (T, 3, 224, 224)
        frame_tensor = frame_tensor.unsqueeze(0)       # (1, T, 3, 224, 224)
        with torch.no_grad():
            feat = self.visual_extractor(frame_tensor)  # (1, 512)
        return feat.squeeze(0).cpu().numpy()

    def extract_audio_features(self, video_path: str) -> np.ndarray:
        audio_path = self.extract_audio(video_path)
        if audio_path:
            feat = self.audio_extractor.extract(audio_path)
            try:
                os.remove(audio_path)
            except Exception:
                pass
            return feat
        return np.zeros(self.AUDIO_DIM)

    def extract_text_features(self, video_path: str) -> np.ndarray:
        audio_path = self.extract_audio(video_path)
        if audio_path:
            # Resample to 16kHz for Wav2Vec2
            text = self.transcriber.transcribe(audio_path)
            try:
                os.remove(audio_path)
            except Exception:
                pass
        else:
            text = ""
        with torch.no_grad():
            feat = self.text_extractor(text)
        return feat.cpu().numpy()

    def extract_all_features(self, video_path: str) -> np.ndarray:
        """Extract and concatenate all modality features → (1293,)"""
        v_feat = self.extract_visual_features(video_path)     # (512,)
        a_feat = self.extract_audio_features(video_path)      # (13,)
        t_feat = self.extract_text_features(video_path)       # (768,)
        combined = np.concatenate([v_feat, a_feat, t_feat])   # (1293,)
        return combined

    # ── Training ──

    def fit(self, feature_matrix: np.ndarray, labels: np.ndarray,
            n_estimators: int = 100) -> None:
        """
        Train Random Forest on pre-extracted features.
        feature_matrix: (N, 1293)
        labels:         (N,) — 0=real, 1=fake
        """
        X_scaled = self.scaler.fit_transform(feature_matrix)
        self.rf_model = RandomForestClassifier(
            n_estimators=n_estimators,
            criterion='gini',
            max_depth=None,
            random_state=42,
            n_jobs=-1
        )
        self.rf_model.fit(X_scaled, labels)

    def save(self, model_path: str, scaler_path: str) -> None:
        if self.rf_model:
            joblib.dump(self.rf_model, model_path)
        joblib.dump(self.scaler, scaler_path)

    # ── Inference ──

    def predict(self, video_path: str) -> Dict[str, Any]:
        """
        Full inference pipeline for a single video.
        Returns dict with prediction, confidence, feature importance.
        """
        features = self.extract_all_features(video_path)
        features_scaled = self.scaler.transform(features.reshape(1, -1))

        if self.rf_model is None:
            # Fallback: heuristic based on visual features
            v_std = features[:self.VISUAL_DIM].std()
            prob_fake = float(np.clip(v_std / 10.0, 0, 1))
            return {
                'prediction': 'FAKE' if prob_fake > 0.5 else 'REAL',
                'probability_fake': prob_fake,
                'confidence': abs(prob_fake - 0.5) * 2,
                'model_available': False
            }

        probs = self.rf_model.predict_proba(features_scaled)[0]
        pred_class = int(self.rf_model.predict(features_scaled)[0])
        prob_fake = float(probs[1]) if len(probs) > 1 else float(probs[0])

        # Feature importance breakdown by modality
        if hasattr(self.rf_model, 'feature_importances_'):
            fi = self.rf_model.feature_importances_
            visual_importance = fi[:self.VISUAL_DIM].sum()
            audio_importance = fi[self.VISUAL_DIM:self.VISUAL_DIM + self.AUDIO_DIM].sum()
            text_importance = fi[self.VISUAL_DIM + self.AUDIO_DIM:].sum()
        else:
            visual_importance = audio_importance = text_importance = 0.0

        return {
            'prediction': 'FAKE' if pred_class == 1 else 'REAL',
            'probability_fake': prob_fake,
            'probability_real': 1.0 - prob_fake,
            'confidence': float(max(probs)),
            'modality_importance': {
                'visual': float(visual_importance),
                'audio': float(audio_importance),
                'text': float(text_importance)
            },
            'raw_features_shape': features.shape,
            'model_available': True
        }

    def predict_image(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Single image deepfake detection (for provenance integration).
        image: (H, W, 3) BGR numpy array
        """
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image_rgb, (224, 224))
        frame = image_resized.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        frame = (frame - mean) / std
        frame_tensor = torch.from_numpy(frame.transpose(2, 0, 1)).float().unsqueeze(0)
        with torch.no_grad():
            v_feat = self.visual_extractor(frame_tensor.to(self.device))
        v_np = v_feat.squeeze(0).cpu().numpy()
        # Pad with zeros for audio/text
        features = np.concatenate([v_np,
                                   np.zeros(self.AUDIO_DIM),
                                   np.zeros(self.TEXT_DIM)])
        if self.rf_model is not None:
            features_scaled = self.scaler.transform(features.reshape(1, -1))
            probs = self.rf_model.predict_proba(features_scaled)[0]
            pred = int(self.rf_model.predict(features_scaled)[0])
            return {
                'prediction': 'FAKE' if pred == 1 else 'REAL',
                'probability_fake': float(probs[1]) if len(probs) > 1 else float(probs[0]),
                'confidence': float(max(probs))
            }
        # Heuristic without model
        return {'prediction': 'UNKNOWN', 'probability_fake': 0.5, 'confidence': 0.0}


# ─────────────────────────────────────────────
# Provenance-Conditioned Deepfake Scoring (Novelty #4)
# ─────────────────────────────────────────────

class ProvenanceConditionedScoring:
    """
    NOVELTY #4: Weight deepfake detection scores by provenance position.
    
    Intuition: Images closer to the root (ancestors) are more trustworthy
    since deepfake generation typically creates offspring, not ancestors.
    This penalizes leaf-node images more heavily.
    """

    def compute_topological_weights(self,
                                    adj_directed: np.ndarray) -> np.ndarray:
        """
        Compute weights based on topological depth.
        adj_directed[i,j] = 1 means i → j (i is ancestor of j)
        Returns: weights (N,) — higher for ancestors (roots), lower for leaves
        """
        N = adj_directed.shape[0]
        in_degree = adj_directed.sum(axis=0)   # (N,) — num parents
        out_degree = adj_directed.sum(axis=1)  # (N,) — num children
        # Depth proxy: nodes with no parents = roots (weight 1.0)
        # Nodes deeper in tree get lower trust weight
        depth = np.zeros(N)
        # BFS-style depth estimation
        visited = set()
        queue = [i for i in range(N) if in_degree[i] == 0]
        for node in queue:
            depth[node] = 0
            visited.add(node)
        # Propagate depth
        changed = True
        while changed:
            changed = False
            for i in range(N):
                children = np.where(adj_directed[i] > 0)[0]
                for j in children:
                    if depth[j] < depth[i] + 1:
                        depth[j] = depth[i] + 1
                        changed = True
        max_depth = max(depth.max(), 1)
        # Sigmoid-inverted weight: ancestors get higher weight
        weights = 1.0 / (1.0 + depth / max_depth)
        return weights

    def apply(self, deepfake_scores: np.ndarray,
              adj_directed: np.ndarray) -> np.ndarray:
        """
        Adjust deepfake scores by provenance position.
        deepfake_scores: (N,) — probability of being fake
        adj_directed:    (N, N) — directed provenance adjacency
        Returns: adjusted_scores (N,)
        """
        weights = self.compute_topological_weights(adj_directed)
        # Boost deepfake scores for leaf nodes (they are more likely manipulated)
        leaf_boost = 1.0 - weights  # higher for leaves
        adjusted = deepfake_scores + 0.2 * leaf_boost
        return np.clip(adjusted, 0, 1)