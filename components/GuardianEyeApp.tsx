import { useState, useCallback, useRef } from "react";

//  Integrates provenance graph analysis + deepfake detection

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

// ── Types ────────────────────────────────────────────────────────────────────

interface ProvenanceNode {
  id: number;
  imageName: string;
  isRoot: boolean;
  isLeaf: boolean;
  manipulationScore: number;
  deepfakeScore: number;
  adjustedScore: number;
  clusterIdx: number;
}

interface ProvenanceEdge {
  source: number;
  target: number;
  confidence: number;
  uncertainty: number;
}

interface AnalysisResult {
  caseId: string;
  numImages: number;
  nodes: ProvenanceNode[];
  edges: ProvenanceEdge[];
  summary: string;
  highRiskImages: number[];
}

// ── Graph Canvas (simple SVG renderer) ─────────────────────────────────────

function ProvenanceGraphCanvas({ result }: { result: AnalysisResult }) {
  const W = 700, H = 400, R = 28;
  const N = result.nodes.length;

  // Simple circular layout
  const positions = result.nodes.map((_, i) => ({
    x: W / 2 + Math.cos((2 * Math.PI * i) / N) * (W / 2 - 60),
    y: H / 2 + Math.sin((2 * Math.PI * i) / N) * (H / 2 - 60),
  }));

  const nodeColor = (node: ProvenanceNode) => {
    const s = node.adjustedScore;
    if (s > 0.65) return "#ef4444";
    if (s > 0.4) return "#f59e0b";
    return "#22c55e";
  };

  return (
    <svg width="100%" viewBox={`0 0 ${W} ${H}`} className="rounded-xl border border-gray-700 bg-gray-900">
      {/* Edges */}
      {result.edges.map((e, i) => {
        const s = positions[e.source];
        const t = positions[e.target];
        if (!s || !t) return null;
        const dx = t.x - s.x, dy = t.y - s.y;
        const len = Math.sqrt(dx * dx + dy * dy);
        const ux = dx / len, uy = dy / len;
        const x1 = s.x + ux * R, y1 = s.y + uy * R;
        const x2 = t.x - ux * R, y2 = t.y - uy * R;
        return (
          <g key={i}>
            <line
              x1={x1} y1={y1} x2={x2} y2={y2}
              stroke={`rgba(148,163,184,${0.3 + e.confidence * 0.7})`}
              strokeWidth={1 + e.confidence * 3}
              markerEnd="url(#arrow)"
            />
          </g>
        );
      })}
      {/* Arrow marker */}
      <defs>
        <marker id="arrow" viewBox="0 0 10 10" refX="5" refY="5"
          markerWidth="6" markerHeight="6" orient="auto-start-reverse">
          <path d="M 0 0 L 10 5 L 0 10 z" fill="#94a3b8" />
        </marker>
      </defs>
      {/* Nodes */}
      {result.nodes.map((node, i) => {
        const p = positions[i];
        return (
          <g key={i} className="cursor-pointer">
            {node.isRoot && (
              <circle cx={p.x} cy={p.y} r={R + 5}
                fill="none" stroke="#fbbf24" strokeWidth={3} strokeDasharray="4,2" />
            )}
            <circle cx={p.x} cy={p.y} r={R}
              fill={nodeColor(node)} opacity={0.9}
              stroke={node.isRoot ? "#fbbf24" : "#1e293b"} strokeWidth={2} />
            <text x={p.x} y={p.y - 6} textAnchor="middle"
              fill="white" fontSize={10} fontWeight="bold">
              Im{i}
            </text>
            <text x={p.x} y={p.y + 8} textAnchor="middle"
              fill="white" fontSize={9}>
              {(node.adjustedScore * 100).toFixed(0)}%
            </text>
            {node.adjustedScore > 0.65 && (
              <text x={p.x} y={p.y + 20} textAnchor="middle" fill="#fca5a5" fontSize={9}>
                ⚠ FAKE
              </text>
            )}
          </g>
        );
      })}
      {/* Legend */}
      {[["#22c55e","Real (<40%)"],["#f59e0b","Uncertain (40-65%)"],["#ef4444","Fake (>65%)"],["#fbbf24","Root"]].map(([c,l],i) => (
        <g key={i} transform={`translate(10, ${10 + i * 18})`}>
          <rect width={12} height={12} fill={c} rx={2} />
          <text x={16} y={10} fill="#94a3b8" fontSize={11}>{l}</text>
        </g>
      ))}
    </svg>
  );
}

// ── Image Card ───────────────────────────────────────────────────────────────

function ImageCard({ node, imgSrc }: { node: ProvenanceNode; imgSrc: string }) {
  const risk = node.adjustedScore > 0.65 ? "FAKE" : node.adjustedScore > 0.4 ? "UNCERTAIN" : "REAL";
  const riskColor = { FAKE: "bg-red-500/20 border-red-500 text-red-400",
                      UNCERTAIN: "bg-amber-500/20 border-amber-500 text-amber-400",
                      REAL: "bg-green-500/20 border-green-500 text-green-400" }[risk];
  return (
    <div className={`rounded-xl border p-3 ${riskColor} flex flex-col gap-2`}>
      <div className="relative">
        <img src={imgSrc} alt={`Image ${node.id}`}
          className="w-full h-28 object-cover rounded-lg" />
        {node.isRoot && (
          <span className="absolute top-1 left-1 bg-amber-500 text-black text-[10px] font-bold px-1.5 py-0.5 rounded">
            ROOT
          </span>
        )}
        {node.isLeaf && (
          <span className="absolute top-1 right-1 bg-slate-500 text-white text-[10px] font-bold px-1.5 py-0.5 rounded">
            LEAF
          </span>
        )}
      </div>
      <div className="text-xs font-semibold">Image {node.id}</div>
      <div className="flex items-center justify-between text-xs">
        <span>Deepfake</span>
        <div className="flex items-center gap-1">
          <div className="w-16 h-1.5 rounded-full bg-gray-700">
            <div className="h-1.5 rounded-full bg-current"
              style={{ width: `${node.adjustedScore * 100}%` }} />
          </div>
          <span className="font-bold">{(node.adjustedScore * 100).toFixed(0)}%</span>
        </div>
      </div>
      <div className="text-xs">
        <span className={`px-2 py-0.5 rounded-full border text-[10px] font-bold ${riskColor}`}>
          {risk}
        </span>
      </div>
    </div>
  );
}

// ── Main Component ───────────────────────────────────────────────────────────

export default function GuardianEyeApp() {
  const [files, setFiles] = useState<File[]>([]);
  const [previews, setPreviews] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<"graph" | "images" | "json">("graph");
  const inputRef = useRef<HTMLInputElement>(null);

  const handleFiles = useCallback((newFiles: File[]) => {
    const imgFiles = newFiles.filter(f => f.type.startsWith("image/"));
    setFiles(imgFiles);
    setPreviews(imgFiles.map(f => URL.createObjectURL(f)));
    setResult(null);
    setError(null);
  }, []);

  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    handleFiles(Array.from(e.dataTransfer.files));
  }, [handleFiles]);

  const analyze = async () => {
    if (files.length < 2) {
      setError("Please upload at least 2 images for provenance analysis.");
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const form = new FormData();
      files.forEach(f => form.append("images", f));
      const res = await fetch(`${API_BASE}/analyze/combined`, {
        method: "POST",
        body: form,
      });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "Analysis failed");
      }
      const data = await res.json();

      // Transform API response to UI model
      const N = data.provenance.num_images;
      const adjD: number[][] = data.provenance.directed_graph;
      const inDeg = Array(N).fill(0);
      const outDeg = Array(N).fill(0);
      for (let i = 0; i < N; i++) {
        for (let j = 0; j < N; j++) {
          if (adjD[i]?.[j] > 0) { outDeg[i]++; inDeg[j]++; }
        }
      }
      const clusterAssign: number[][] = data.provenance.cluster_assignments;
      const nodes: ProvenanceNode[] = Array.from({ length: N }, (_, i) => ({
        id: i,
        imageName: files[i]?.name || `image_${i}`,
        isRoot: inDeg[i] === 0,
        isLeaf: outDeg[i] === 0,
        manipulationScore: data.provenance.frequency_scores[i] ?? 0,
        deepfakeScore: data.deepfake_scores[i]?.probability_fake ?? 0.5,
        adjustedScore: data.provenance_adjusted_scores[i] ?? 0.5,
        clusterIdx: clusterAssign[i] ? clusterAssign[i].indexOf(Math.max(...clusterAssign[i])) : 0,
      }));

      const edges: ProvenanceEdge[] = [];
      const ep: number[][] = data.provenance.edge_probabilities;
      const eu: number[][] = data.provenance.edge_uncertainties;
      for (let i = 0; i < N; i++) {
        for (let j = 0; j < N; j++) {
          if (adjD[i]?.[j] > 0) {
            edges.push({ source: i, target: j,
                         confidence: ep[i]?.[j] ?? 0.5,
                         uncertainty: eu[i]?.[j] ?? 0 });
          }
        }
      }
      setResult({
        caseId: data.case_id,
        numImages: N,
        nodes,
        edges,
        summary: data.analysis_summary,
        highRiskImages: data.high_risk_images,
      });
    } catch (e: any) {
      setError(e.message || "Unknown error");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-950 text-gray-100 font-sans">
      {/* Header */}
      <header className="border-b border-gray-800 px-6 py-4 flex items-center gap-4">
        <div className="w-9 h-9 rounded-lg bg-gradient-to-br from-blue-500 to-violet-600 flex items-center justify-center text-white font-bold text-lg">
          G
        </div>
        <div>
          <h1 className="text-lg font-bold text-white">Guardian Eye</h1>
          <p className="text-xs text-gray-400">
            Image Provenance Analysis × Multi-Modal Deepfake Detection
          </p>
        </div>
        <div className="ml-auto flex gap-2 text-xs text-gray-500">
          <span className="bg-blue-500/10 border border-blue-500/30 text-blue-400 px-2 py-1 rounded">
            ViT-LoRA
          </span>
          <span className="bg-violet-500/10 border border-violet-500/30 text-violet-400 px-2 py-1 rounded">
            Graph Encoding
          </span>
          <span className="bg-rose-500/10 border border-rose-500/30 text-rose-400 px-2 py-1 rounded">
            Guardian Eye
          </span>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-6 py-8 space-y-8">
        {/* Upload Zone */}
        <section
          onDrop={onDrop}
          onDragOver={e => e.preventDefault()}
          onClick={() => inputRef.current?.click()}
          className="border-2 border-dashed border-gray-700 hover:border-blue-500 rounded-2xl p-10 text-center cursor-pointer transition-colors"
        >
          <input ref={inputRef} type="file" multiple accept="image/*"
            className="hidden"
            onChange={e => handleFiles(Array.from(e.target.files || []))} />
          <div className="text-5xl mb-3">🖼️</div>
          <p className="text-gray-300 font-semibold text-lg">
            Drop related images here (3–50 images)
          </p>
          <p className="text-gray-500 text-sm mt-1">
            Upload a set of semantically similar images to trace their manipulation history
          </p>
          {files.length > 0 && (
            <p className="mt-3 text-blue-400 font-medium">
              {files.length} image{files.length !== 1 ? "s" : ""} selected
            </p>
          )}
        </section>

        {/* Previews */}
        {previews.length > 0 && (
          <div className="grid grid-cols-4 sm:grid-cols-6 md:grid-cols-8 gap-2">
            {previews.map((src, i) => (
              <div key={i} className="relative aspect-square rounded-lg overflow-hidden border border-gray-700">
                <img src={src} alt="" className="w-full h-full object-cover" />
                <span className="absolute bottom-0 right-0 bg-black/70 text-white text-[10px] px-1">
                  {i}
                </span>
              </div>
            ))}
          </div>
        )}

        {/* Analyze Button */}
        {files.length >= 2 && !result && (
          <div className="flex justify-center">
            <button
              onClick={analyze}
              disabled={loading}
              className="px-8 py-3 rounded-xl bg-gradient-to-r from-blue-600 to-violet-600 text-white font-bold text-base hover:opacity-90 disabled:opacity-50 transition flex items-center gap-2"
            >
              {loading ? (
                <>
                  <span className="animate-spin">⟳</span> Analyzing…
                </>
              ) : (
                <>🔍 Analyze Provenance + Deepfake</>
              )}
            </button>
          </div>
        )}

        {/* Error */}
        {error && (
          <div className="rounded-xl border border-red-500/50 bg-red-500/10 p-4 text-red-400 text-sm">
            ❌ {error}
          </div>
        )}

        {/* Results */}
        {result && (
          <section className="space-y-6">
            {/* Summary Banner */}
            <div className={`rounded-xl border p-5 ${
              result.highRiskImages.length > 0
                ? "border-red-500/50 bg-red-500/10"
                : "border-green-500/50 bg-green-500/10"
            }`}>
              <div className="flex items-start gap-3">
                <span className="text-2xl mt-0.5">
                  {result.highRiskImages.length > 0 ? "⚠️" : "✅"}
                </span>
                <div>
                  <div className="font-bold text-white mb-1">
                    {result.highRiskImages.length > 0
                      ? `${result.highRiskImages.length} High-Risk Image(s) Detected`
                      : "No High-Risk Manipulations Detected"}
                  </div>
                  <p className="text-sm text-gray-300">{result.summary}</p>
                  <p className="text-xs text-gray-500 mt-2">Case ID: {result.caseId}</p>
                </div>
              </div>
            </div>

            {/* Tabs */}
            <div className="flex gap-1 border-b border-gray-800">
              {(["graph", "images", "json"] as const).map(tab => (
                <button key={tab}
                  onClick={() => setActiveTab(tab)}
                  className={`px-5 py-2.5 text-sm font-medium rounded-t-lg transition ${
                    activeTab === tab
                      ? "bg-gray-800 text-white"
                      : "text-gray-400 hover:text-gray-200"
                  }`}>
                  {tab === "graph" ? "📊 Provenance Graph"
                    : tab === "images" ? "🖼️ Image Analysis"
                    : "{ } JSON"}
                </button>
              ))}
            </div>

            {activeTab === "graph" && (
              <div className="space-y-4">
                <ProvenanceGraphCanvas result={result} />
                {/* Edge Table */}
                <div className="rounded-xl border border-gray-700 bg-gray-900 overflow-hidden">
                  <div className="px-4 py-3 border-b border-gray-700 text-sm font-semibold text-gray-300">
                    Directed Provenance Edges
                  </div>
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="bg-gray-800 text-gray-400 text-xs">
                        <th className="px-4 py-2 text-left">Source</th>
                        <th className="px-4 py-2 text-left">Target</th>
                        <th className="px-4 py-2 text-left">Confidence</th>
                        <th className="px-4 py-2 text-left">Uncertainty</th>
                      </tr>
                    </thead>
                    <tbody>
                      {result.edges.map((e, i) => (
                        <tr key={i} className="border-t border-gray-800">
                          <td className="px-4 py-2">Image {e.source}</td>
                          <td className="px-4 py-2">Image {e.target}</td>
                          <td className="px-4 py-2">
                            <div className="flex items-center gap-2">
                              <div className="w-20 h-1.5 rounded-full bg-gray-700">
                                <div className="h-1.5 rounded-full bg-blue-500"
                                  style={{ width: `${e.confidence * 100}%` }} />
                              </div>
                              <span>{(e.confidence * 100).toFixed(0)}%</span>
                            </div>
                          </td>
                          <td className="px-4 py-2 text-gray-400">
                            ±{e.uncertainty.toFixed(3)}
                          </td>
                        </tr>
                      ))}
                      {result.edges.length === 0 && (
                        <tr>
                          <td colSpan={4} className="px-4 py-6 text-center text-gray-500">
                            No directed edges found
                          </td>
                        </tr>
                      )}
                    </tbody>
                  </table>
                </div>
              </div>
            )}

            {activeTab === "images" && (
              <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-4">
                {result.nodes.map((node, i) => (
                  <ImageCard key={i} node={node} imgSrc={previews[i] || ""} />
                ))}
              </div>
            )}

            {activeTab === "json" && (
              <pre className="rounded-xl border border-gray-700 bg-gray-900 p-4 text-xs text-green-400 overflow-auto max-h-96">
                {JSON.stringify({
                  case_id: result.caseId,
                  num_images: result.numImages,
                  high_risk_images: result.highRiskImages,
                  nodes: result.nodes,
                  edges: result.edges,
                }, null, 2)}
              </pre>
            )}
          </section>
        )}

        {/* Architecture Info */}
        <section className="border border-gray-800 rounded-2xl p-6">
          <h2 className="text-sm font-bold text-gray-300 mb-4 uppercase tracking-wide">
            System Architecture
          </h2>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-xs text-gray-400">
            {[
              { icon: "🔬", title: "ViT-LoRA Encoder", desc: "Pre-trained ViT-B/16 with Low-Rank Adaptation (r=16) for patch embeddings" },
              { icon: "🕸️", title: "Graph Encoder", desc: "Graph Structure Masked Attention with virtual source/target nodes for direction" },
              { icon: "🎵", title: "Multi-Modal Fusion", desc: "ResNet-18 + MFCC (13) + BERT-768 → 1293-dim → Random Forest" },
              { icon: "⚡", title: "4 Novel Contributions", desc: "DCT consistency · MC dropout uncertainty · Hierarchical clustering · Provenance-conditioned scoring" },
            ].map(({ icon, title, desc }) => (
              <div key={title} className="rounded-lg bg-gray-900 border border-gray-700 p-4">
                <div className="text-2xl mb-2">{icon}</div>
                <div className="font-semibold text-gray-200 mb-1">{title}</div>
                <div>{desc}</div>
              </div>
            ))}
          </div>
        </section>
      </main>
    </div>
  );
}