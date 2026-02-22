import os
import sys
import argparse
import json
import numpy as np
import torch
import cv2
from PIL import Image
from torchvision import transforms
from typing import List, Dict, Optional
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.provenance_model import ProvenanceAnalysisModel, kruskal_mst
from models.deepfake_detector import (
    GuardianEyeDeepfakeDetector,
    ProvenanceConditionedScoring
)


TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


def load_images(image_paths: List[str]) -> torch.Tensor:
    """Load and preprocess a list of images → (N, 3, 224, 224)."""
    tensors = []
    for path in image_paths:
        img = Image.open(path).convert('RGB')
        tensors.append(TRANSFORM(img))
    return torch.stack(tensors)


def run_provenance_inference(model: ProvenanceAnalysisModel,
                              image_paths: List[str],
                              device: str = 'cpu') -> Dict:
    """
    Run provenance analysis on a set of images.
    Returns comprehensive output dict.
    """
    images = load_images(image_paths).to(device)
    model.eval()
    with torch.no_grad():
        output = model(images, mode='inference')

    # Convert to numpy
    adj_u = output['adj_undirected'].cpu().numpy()
    adj_d = output['adj_directed'].cpu().numpy()
    dir_mat = output['direction_matrix'].cpu().numpy()
    freq_scores = output['freq_scores'].cpu().numpy()
    edge_probs = output['edge_probabilities'].squeeze(0).cpu().numpy()
    edge_vars = output['edge_uncertainties'].squeeze(0).cpu().numpy()
    cluster_assign = output['cluster_assignments'].squeeze(0).cpu().numpy()

    return {
        'image_paths': image_paths,
        'num_images': len(image_paths),
        'adj_undirected': adj_u,
        'adj_directed': adj_d,
        'direction_matrix': dir_mat,
        'freq_manipulation_scores': freq_scores.tolist(),
        'edge_probabilities': edge_probs,
        'edge_uncertainties': edge_vars,
        'cluster_assignments': cluster_assign,
    }


def visualize_provenance_graph(result: Dict,
                                output_path: str = 'provenance_graph.png',
                                df_scores: Optional[np.ndarray] = None) -> None:
    """
    Visualize the directed provenance graph.
    - Green nodes: low deepfake probability
    - Red nodes: high deepfake probability
    - Edge thickness: edge confidence
    """
    adj_d = result['adj_directed']
    freq_scores = np.array(result['freq_manipulation_scores'])
    N = adj_d.shape[0]
    image_paths = result['image_paths']
    edge_probs = result['edge_probabilities']

    G = nx.DiGraph()
    G.add_nodes_from(range(N))
    for i in range(N):
        for j in range(N):
            if adj_d[i, j] > 0:
                G.add_edge(i, j, weight=float(edge_probs[i, j]))

    # Node colors based on deepfake score
    scores = df_scores if df_scores is not None else freq_scores
    scores = np.clip(scores, 0, 1)
    node_colors = []
    for s in scores:
        # Red channel increases with score, green decreases
        node_colors.append((float(s), float(1 - s), 0.2))

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    # ── Left: Graph visualization ──
    ax1 = axes[0]
    try:
        pos = nx.spring_layout(G, seed=42, k=2.0)
    except Exception:
        pos = {i: (i % 5, i // 5) for i in range(N)}

    # Draw edges with varying thickness
    edges = G.edges(data=True)
    edge_weights = [e[2].get('weight', 0.5) * 4 for e in edges]
    nx.draw_networkx_edges(G, pos, ax=ax1, arrows=True,
                           arrowsize=20, width=edge_weights,
                           edge_color='#555555', connectionstyle='arc3,rad=0.1')
    nx.draw_networkx_nodes(G, pos, ax=ax1, node_color=node_colors,
                           node_size=600, alpha=0.9)
    # Node labels with image index
    labels = {i: f"Im{i}\n{scores[i]:.2f}" for i in range(N)}
    nx.draw_networkx_labels(G, pos, labels=labels, ax=ax1, font_size=8)

    # In-degree == 0 → root marker
    roots = [i for i in range(N) if adj_d[:, i].sum() == 0]
    if roots:
        nx.draw_networkx_nodes(G, pos, nodelist=roots, ax=ax1,
                               node_color='gold', node_size=800,
                               edgecolors='black', linewidths=2)

    ax1.set_title('Directed Provenance Graph\n(Gold = Root, Color = Manipulation Risk)',
                  fontsize=12, fontweight='bold')

    # Legend
    patches = [
        mpatches.Patch(color='gold', label='Root image'),
        mpatches.Patch(color='red', label='High manipulation risk'),
        mpatches.Patch(color='green', label='Low manipulation risk'),
    ]
    ax1.legend(handles=patches, loc='lower left', fontsize=9)
    ax1.axis('off')

    # ── Right: Cluster heatmap ──
    ax2 = axes[1]
    cluster_assign = result['cluster_assignments']
    if cluster_assign.ndim == 2 and cluster_assign.shape[0] == N:
        im = ax2.imshow(cluster_assign, cmap='YlOrRd', aspect='auto')
        ax2.set_xlabel('Cluster Index', fontsize=11)
        ax2.set_ylabel('Image Index', fontsize=11)
        ax2.set_title('Hierarchical Cluster Assignments\n(NOVELTY #3)', fontsize=12, fontweight='bold')
        plt.colorbar(im, ax=ax2)
        ax2.set_xticks(range(cluster_assign.shape[1]))
        ax2.set_yticks(range(N))
        ax2.set_yticklabels([os.path.basename(p)[:15] for p in image_paths[:N]],
                            fontsize=7)
    else:
        ax2.text(0.5, 0.5, 'Cluster data unavailable', ha='center', va='center',
                 transform=ax2.transAxes)
        ax2.axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Graph saved to: {output_path}")


def print_report(result: Dict,
                 deepfake_scores: Optional[np.ndarray] = None,
                 adj_scores: Optional[np.ndarray] = None) -> None:
    """Print a structured analysis report to stdout."""
    N = result['num_images']
    adj_d = result['adj_directed']
    in_deg = adj_d.sum(axis=0)
    roots = [i for i in range(N) if in_deg[i] == 0]
    leaves = [i for i in range(N) if adj_d[i].sum() == 0]
    freq = np.array(result['freq_manipulation_scores'])

    print("\n" + "=" * 60)
    print("  GUARDIAN EYE + PROVENANCE ANALYSIS REPORT")
    print("=" * 60)
    print(f"  Images analyzed: {N}")
    print(f"  Root images:     {roots}")
    print(f"  Leaf images:     {leaves}")
    print()
    print("  ── Manipulation Risk (frequency domain) ──")
    for i in range(N):
        name = os.path.basename(result['image_paths'][i])[:30]
        bar = '█' * int(freq[i] * 20)
        risk = "⚠️ HIGH" if freq[i] > 0.6 else ("  MED" if freq[i] > 0.4 else "  LOW")
        print(f"  [{i:2d}] {name:<30} {bar:<20} {freq[i]:.3f} {risk}")

    if adj_scores is not None:
        print()
        print("  ── Provenance-Adjusted Deepfake Scores ──")
        for i in range(N):
            name = os.path.basename(result['image_paths'][i])[:30]
            bar = '█' * int(adj_scores[i] * 20)
            risk = "⚠️ FAKE" if adj_scores[i] > 0.65 else "  REAL"
            print(f"  [{i:2d}] {name:<30} {bar:<20} {adj_scores[i]:.3f} {risk}")

    print()
    print("  ── Provenance Graph Edges ──")
    for i in range(N):
        for j in range(N):
            if adj_d[i, j] > 0:
                ep = result['edge_probabilities'][i, j]
                eu = result['edge_uncertainties'][i, j]
                print(f"  Image[{i}] ──▶ Image[{j}]  "
                      f"conf={ep:.3f} ±{eu:.3f}")
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--images', nargs='+', required=True,
                        help='Paths to input images')
    parser.add_argument('--weights', type=str, default=None,
                        help='Path to provenance model weights')
    parser.add_argument('--deepfake_model', type=str, default=None)
    parser.add_argument('--deepfake_scaler', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default='outputs')
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load models ──
    prov_model = ProvenanceAnalysisModel(
        lora_rank=16, hidden_dim=768, num_graph_layers=3, num_clusters=8
    ).to(args.device)
    if args.weights and os.path.exists(args.weights):
        state = torch.load(args.weights, map_location=args.device)
        prov_model.load_state_dict(state, strict=False)
        print(f"Loaded provenance weights from: {args.weights}")
    prov_model.eval()

    detector = GuardianEyeDeepfakeDetector(
        model_path=args.deepfake_model,
        scaler_path=args.deepfake_scaler,
        device=args.device
    )
    scorer = ProvenanceConditionedScoring()

    # ── Run provenance analysis ──
    print(f"Analyzing {len(args.images)} images...")
    prov_result = run_provenance_inference(prov_model, args.images, args.device)

    # ── Run deepfake detection ──
    raw_df_scores = []
    for img_path in args.images:
        img = cv2.imread(img_path)
        if img is not None:
            r = detector.predict_image(img)
            raw_df_scores.append(r.get('probability_fake', 0.5))
        else:
            raw_df_scores.append(0.5)

    raw_scores_np = np.array(raw_df_scores)
    adj_scores = scorer.apply(raw_scores_np, prov_result['adj_directed'])

    # ── Report + Visualization ──
    print_report(prov_result, raw_scores_np, adj_scores)

    vis_path = os.path.join(args.output_dir, 'provenance_graph.png')
    visualize_provenance_graph(prov_result, vis_path, adj_scores)

    # Save JSON result
    result_save = {
        'image_paths': prov_result['image_paths'],
        'num_images': prov_result['num_images'],
        'adj_directed': prov_result['adj_directed'].tolist(),
        'adj_undirected': prov_result['adj_undirected'].tolist(),
        'freq_manipulation_scores': prov_result['freq_manipulation_scores'],
        'raw_deepfake_scores': raw_scores_np.tolist(),
        'provenance_adjusted_scores': adj_scores.tolist(),
        'high_risk_images': [i for i, s in enumerate(adj_scores) if s > 0.65]
    }
    json_path = os.path.join(args.output_dir, 'provenance_result.json')
    with open(json_path, 'w') as f:
        json.dump(result_save, f, indent=2)
    print(f"Results saved to: {json_path}")


if __name__ == '__main__':
    main()