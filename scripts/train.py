import argparse
import os
import sys
import json
import random
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from typing import List, Dict, Tuple, Optional
import cv2
import logging

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.provenance_model import ProvenanceAnalysisModel, kruskal_mst

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────

class ProvenanceDataset(Dataset):
    """
    Loads provenance graph cases.
    Each case is a directory with:
      - images/: PNG/JPG files (the image set)
      - graph.json: {
            "edges": [[src_idx, tgt_idx], ...],
            "manipulation_labels": [0, 1, 0, ...]  (optional)
        }
    """

    IMAGE_SIZE = 224
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    def __init__(self, data_root: str, split: str = 'train', max_images: int = 20):
        self.data_root = data_root
        self.split = split
        self.max_images = max_images
        self.cases = self._load_cases()
        logger.info(f"Loaded {len(self.cases)} cases for split={split}")

    def _load_cases(self) -> List[str]:
        split_file = os.path.join(self.data_root, f'{self.split}_cases.txt')
        if os.path.exists(split_file):
            with open(split_file) as f:
                return [l.strip() for l in f if l.strip()]
        # Fallback: list all case directories
        all_cases = []
        for item in sorted(os.listdir(self.data_root)):
            case_dir = os.path.join(self.data_root, item)
            if os.path.isdir(case_dir) and os.path.exists(
                    os.path.join(case_dir, 'graph.json')):
                all_cases.append(case_dir)
        # 70/30 split
        random.seed(42)
        random.shuffle(all_cases)
        cutoff = int(0.7 * len(all_cases))
        return all_cases[:cutoff] if self.split == 'train' else all_cases[cutoff:]

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, idx: int) -> Dict:
        case_dir = self.cases[idx]

        # Load graph
        with open(os.path.join(case_dir, 'graph.json')) as f:
            graph_info = json.load(f)
        edges = graph_info.get('edges', [])
        manip_labels = graph_info.get('manipulation_labels', None)

        # Load images
        img_dir = os.path.join(case_dir, 'images')
        img_files = sorted([
            f for f in os.listdir(img_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])[:self.max_images]

        images = []
        for fname in img_files:
            img = Image.open(os.path.join(img_dir, fname)).convert('RGB')
            images.append(self.transform(img))

        N = len(images)
        image_tensor = torch.stack(images)  # (N, 3, 224, 224)

        # Build adjacency matrices
        adj_directed = np.zeros((N, N), dtype=np.float32)
        for (src, tgt) in edges:
            if src < N and tgt < N:
                adj_directed[src, tgt] = 1.0
        adj_undirected = np.clip(adj_directed + adj_directed.T, 0, 1)

        # Compute path lengths (BFS)
        path_lengths = self._bfs_path_lengths(adj_directed, N)

        # Manipulation labels
        if manip_labels is not None:
            ml = np.array(manip_labels[:N], dtype=np.float32)
        else:
            # Heuristic: leaf nodes (no children) are likely manipulated
            out_degree = adj_directed.sum(axis=1)
            ml = (out_degree == 0).astype(np.float32)

        return {
            'images': image_tensor,
            'adj_directed': torch.from_numpy(adj_directed),
            'adj_undirected': torch.from_numpy(adj_undirected),
            'path_lengths': torch.from_numpy(path_lengths),
            'manip_labels': torch.from_numpy(ml),
            'num_images': N,
            'case_dir': case_dir
        }

    def _bfs_path_lengths(self, adj: np.ndarray, n: int) -> np.ndarray:
        """BFS from each node to compute shortest path lengths."""
        lengths = np.zeros((n, n), dtype=np.float32)
        for start in range(n):
            dist = {start: 0}
            queue = [start]
            while queue:
                node = queue.pop(0)
                for nxt in range(n):
                    if adj[node, nxt] > 0 and nxt not in dist:
                        dist[nxt] = dist[node] + 1
                        queue.append(nxt)
            for j, d in dist.items():
                lengths[start, j] = d
        return lengths


def collate_fn(batch):
    """Handle variable-length provenance cases (different N per case)."""
    # Return first item only (single-case training, batch_size=1 recommended)
    return batch[0]


# ─────────────────────────────────────────────
# Data Augmentation for Provenance (from paper §IV-B)
# ─────────────────────────────────────────────

class ProvenanceAugmentation:
    """
    Two augmentation strategies:
      1. Add branches: apply global manipulations to create new leaf nodes
      2. Remove single-ancestor nodes: simplify graph structure
    """

    GLOBAL_OPS = ['brightness', 'contrast', 'blur', 'noise', 'saturation']

    def apply_global_transform(self, image: torch.Tensor,
                               op: str) -> torch.Tensor:
        """Apply a global image transformation."""
        img_np = image.permute(1, 2, 0).numpy()
        img_uint8 = ((img_np * np.array([0.229, 0.224, 0.225]) +
                      np.array([0.485, 0.456, 0.406])) * 255).clip(0, 255).astype(np.uint8)

        if op == 'brightness':
            factor = np.random.uniform(0.5, 1.5)
            img_uint8 = cv2.convertScaleAbs(img_uint8, alpha=factor)
        elif op == 'contrast':
            factor = np.random.uniform(0.5, 1.5)
            img_uint8 = cv2.convertScaleAbs(img_uint8, beta=factor * 10)
        elif op == 'blur':
            ksize = random.choice([3, 5, 7])
            img_uint8 = cv2.GaussianBlur(img_uint8, (ksize, ksize), 0)
        elif op == 'noise':
            noise = np.random.randn(*img_uint8.shape) * 10
            img_uint8 = np.clip(img_uint8.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        elif op == 'saturation':
            hsv = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2HSV).astype(np.float32)
            hsv[:, :, 1] *= np.random.uniform(0.5, 1.5)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
            img_uint8 = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

        # Re-normalize
        img_float = img_uint8.astype(np.float32) / 255.0
        img_norm = (img_float - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        return torch.from_numpy(img_norm.transpose(2, 0, 1)).float()

    def augment(self, sample: Dict, prob: float = 0.5) -> Dict:
        """Apply random augmentation to a provenance sample."""
        if random.random() > prob:
            return sample

        images = sample['images']   # (N, 3, H, W)
        adj_d = sample['adj_directed'].numpy()
        N = sample['num_images']

        # Strategy 1: Add a new branch from a random node
        if N < 15 and random.random() < 0.5:
            src_node = random.randint(0, N - 1)
            op = random.choice(self.GLOBAL_OPS)
            new_img = self.apply_global_transform(images[src_node], op)
            images = torch.cat([images, new_img.unsqueeze(0)], dim=0)
            # Add edge src_node → N (new node)
            new_adj = np.zeros((N + 1, N + 1), dtype=np.float32)
            new_adj[:N, :N] = adj_d
            new_adj[src_node, N] = 1.0
            adj_d = new_adj
            N += 1

        adj_u = np.clip(adj_d + adj_d.T, 0, 1)
        # Recompute path lengths
        dataset = ProvenanceDataset.__new__(ProvenanceDataset)
        path_lengths = dataset._bfs_path_lengths(adj_d, N)

        return {
            'images': images,
            'adj_directed': torch.from_numpy(adj_d),
            'adj_undirected': torch.from_numpy(adj_u),
            'path_lengths': torch.from_numpy(path_lengths),
            'manip_labels': sample['manip_labels'],
            'num_images': N,
            'case_dir': sample['case_dir']
        }


# ─────────────────────────────────────────────
# Evaluation Metrics (VO, EO, VEO)
# ─────────────────────────────────────────────

def compute_metrics(adj_pred: np.ndarray,
                    adj_gt: np.ndarray) -> Dict[str, float]:
    """
    Compute Vertex Overlap (VO), Edge Overlap (EO), Vertex-Edge Overlap (VEO).
    All use F1 score (harmonic mean of precision and recall).
    """
    # For VO: if using MST, all vertices are included → VO = 1.0
    # EO and VEO require edge comparison
    pred_edges = set(zip(*np.where(adj_pred > 0)))
    gt_edges = set(zip(*np.where(adj_gt > 0)))

    n_pred = len(pred_edges)
    n_gt = len(gt_edges)
    n_correct = len(pred_edges & gt_edges)

    if n_pred + n_gt == 0:
        eo = 1.0
    else:
        eo = 2 * n_correct / (n_pred + n_gt)

    # VO: assume all vertices included (since MST always connects all)
    N_pred = adj_pred.shape[0]
    N_gt = adj_gt.shape[0]
    N_common = min(N_pred, N_gt)
    vo = 2 * N_common / (N_pred + N_gt) if (N_pred + N_gt) > 0 else 1.0

    veo_num = 2 * (N_common + n_correct)
    veo_den = N_pred + N_gt + n_pred + n_gt
    veo = veo_num / veo_den if veo_den > 0 else 0.0

    return {'VO': vo, 'EO': eo, 'VEO': veo}


# ─────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────

class ProvenanceTrainer:

    def __init__(self, args):
        self.args = args
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device: {self.device}")

        # Model
        self.model = ProvenanceAnalysisModel(
            lora_rank=args.lora_rank,
            hidden_dim=768,
            num_graph_layers=args.graph_layers,
            num_clusters=args.num_clusters
        ).to(self.device)

        # Optimizer (AdamW, constant LR as per paper)
        self.optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=args.lr,
            weight_decay=1e-4
        )

        # Data
        self.train_dataset = ProvenanceDataset(args.data_root, 'train', args.max_images)
        self.val_dataset = ProvenanceDataset(args.data_root, 'val', args.max_images)
        self.augmenter = ProvenanceAugmentation()

        os.makedirs(args.output_dir, exist_ok=True)

    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        num_cases = 0

        for idx in range(len(self.train_dataset)):
            sample = self.train_dataset[idx]
            # Augment
            sample = self.augmenter.augment(sample, prob=0.5)

            images = sample['images'].to(self.device)
            adj_gt = sample['adj_undirected'].to(self.device)
            path_lengths = sample['path_lengths'].to(self.device)
            dir_adj_gt = sample['adj_directed'].to(self.device)
            manip_labels = sample['manip_labels'].to(self.device)

            self.optimizer.zero_grad()
            output = self.model(
                images,
                adj_gt=adj_gt,
                path_lengths=path_lengths,
                dir_adj_gt=dir_adj_gt,
                manip_labels=manip_labels,
                mode='train'
            )
            loss = output['loss']
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            num_cases += 1

            if idx % 50 == 0:
                logger.info(
                    f"Epoch {epoch} [{idx}/{len(self.train_dataset)}] "
                    f"Loss: {loss.item():.4f} "
                    f"(Lu: {output.get('L_u', 0):.4f}, "
                    f"Ld: {output.get('L_d', 0):.4f})"
                )

        return total_loss / max(num_cases, 1)

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        self.model.eval()
        all_metrics = {'VO': [], 'EO': [], 'VEO': []}

        for idx in range(len(self.val_dataset)):
            sample = self.val_dataset[idx]
            images = sample['images'].to(self.device)
            adj_gt_d = sample['adj_directed'].numpy()

            output = self.model(images, mode='inference')
            adj_pred = output['adj_directed'].cpu().numpy()

            m = compute_metrics(adj_pred, adj_gt_d)
            for k, v in m.items():
                all_metrics[k].append(v)

        return {k: float(np.mean(v)) for k, v in all_metrics.items()}

    def train(self):
        best_veo = 0.0
        history = []

        for epoch in range(1, self.args.epochs + 1):
            train_loss = self.train_epoch(epoch)
            metrics = self.evaluate()
            history.append({'epoch': epoch, 'loss': train_loss, **metrics})

            logger.info(
                f"Epoch {epoch}/{self.args.epochs} | "
                f"Loss: {train_loss:.4f} | "
                f"VO: {metrics['VO']:.4f} | "
                f"EO: {metrics['EO']:.4f} | "
                f"VEO: {metrics['VEO']:.4f}"
            )

            if metrics['VEO'] > best_veo:
                best_veo = metrics['VEO']
                save_path = os.path.join(self.args.output_dir, 'provenance_best.pth')
                torch.save(self.model.state_dict(), save_path)
                logger.info(f"  ✓ Best model saved (VEO={best_veo:.4f})")

        # Save training history
        hist_path = os.path.join(self.args.output_dir, 'training_history.json')
        with open(hist_path, 'w') as f:
            json.dump(history, f, indent=2)
        logger.info(f"Training complete. Best VEO: {best_veo:.4f}")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description='Train Provenance Analysis Model')
    parser.add_argument('--data_root', type=str, required=True,
                        help='Root directory with provenance cases')
    parser.add_argument('--output_dir', type=str, default='weights',
                        help='Directory to save model weights')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--max_images', type=int, default=20,
                        help='Max images per provenance case')
    parser.add_argument('--lora_rank', type=int, default=16)
    parser.add_argument('--graph_layers', type=int, default=3)
    parser.add_argument('--num_clusters', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    trainer = ProvenanceTrainer(args)
    trainer.train()