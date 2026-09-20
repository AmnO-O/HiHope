import React from 'react';
import { PrototypeMetrics, TargetType } from '../types';
import { GitCompare, ArrowRight, Activity, Zap, Compass } from 'lucide-react';

interface PrototypeStreamViewerProps {
  metrics: PrototypeMetrics;
  targetWord: string;
  activeTarget: TargetType;
}

export const PrototypeStreamViewer: React.FC<PrototypeStreamViewerProps> = ({
  metrics,
  targetWord,
  activeTarget,
}) => {
  return (
    <div className="bg-white rounded-xl border border-slate-200/80 shadow-xs p-5 space-y-4">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 pb-3 border-b border-slate-100">
        <div className="flex items-center space-x-2">
          <GitCompare className="w-4 h-4 text-indigo-600" />
          <h3 className="text-xs font-bold uppercase tracking-wider text-slate-800">
            Two-Stream Prototype & Semantic Displacement (<span className="font-mono text-indigo-600">src/prototype_stream.py</span>)
          </h3>
        </div>

        <span className="text-xs font-semibold px-2.5 py-0.5 rounded-full bg-slate-100 text-slate-700">
          Target Lemma: <span className="font-mono font-bold text-indigo-700">"{targetWord || 'none'}"</span>
        </span>
      </div>

      {/* Two Stream Architecture Diagram Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
        {/* Stream 1: Isolated Lexical Prototype */}
        <div className="p-4 rounded-xl bg-blue-50/50 border border-blue-200/70 space-y-2">
          <div className="flex items-center justify-between text-xs font-bold text-blue-900 uppercase">
            <span>Stream 1: Prototype</span>
            <span className="px-1.5 py-0.2 rounded bg-blue-200 text-blue-800 font-mono text-[10px]">
              h_proto
            </span>
          </div>
          <p className="text-xs text-blue-950 font-medium">
            Encodes isolated lemma <span className="font-mono font-bold">"{targetWord}"</span> out-of-context without surrounding syntax.
          </p>
          <div className="pt-1 text-[11px] text-blue-700 font-mono">
            Prototype Norm ||h_proto||: {metrics.protoNorm}
          </div>
        </div>

        {/* Stream 2: Contextual Representation */}
        <div className="p-4 rounded-xl bg-purple-50/50 border border-purple-200/70 space-y-2">
          <div className="flex items-center justify-between text-xs font-bold text-purple-900 uppercase">
            <span>Stream 2: In-Context</span>
            <span className="px-1.5 py-0.2 rounded bg-purple-200 text-purple-800 font-mono text-[10px]">
              h_ctx
            </span>
          </div>
          <p className="text-xs text-purple-950 font-medium">
            Encodes the full context sentence and extracts pooled span embeddings for <span className="font-mono font-bold">{activeTarget}</span>.
          </p>
          <div className="pt-1 text-[11px] text-purple-700 font-mono">
            Context Norm ||h_ctx||: {metrics.ctxNorm}
          </div>
        </div>

        {/* Semantic Shift Cross-Fusion */}
        <div className="p-4 rounded-xl bg-emerald-50/50 border border-emerald-200/70 space-y-2">
          <div className="flex items-center justify-between text-xs font-bold text-emerald-900 uppercase">
            <span>Semantic Shift Cross-Fusion</span>
            <span className="px-1.5 py-0.2 rounded bg-emerald-200 text-emerald-800 font-mono text-[10px]">
              Δh = h_ctx - h_proto
            </span>
          </div>
          <p className="text-xs text-emerald-950 font-medium">
            Constructs 3-token sequence with learned role embeddings: [h_ctx, h_proto, Δh] for cross-attention.
          </p>
          <div className="pt-1 text-[11px] text-emerald-700 font-mono">
            Drift Magnitude ||Δh||: {metrics.semanticDisplacementNorm}
          </div>
        </div>
      </div>

      {/* Metric Gauge & Vector Inspection */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 pt-2">
        <div className="p-4 rounded-xl bg-slate-50 border border-slate-200/80 space-y-3">
          <div className="flex items-center justify-between">
            <span className="text-xs font-bold text-slate-700 uppercase tracking-wider">
              Cosine Similarity cos(h_ctx, h_proto)
            </span>
            <span className="text-base font-bold font-mono text-indigo-700">
              {metrics.cosineSim > 0 ? `+${metrics.cosineSim}` : metrics.cosineSim}
            </span>
          </div>

          {/* Progress bar */}
          <div className="w-full bg-slate-200 h-2.5 rounded-full overflow-hidden">
            <div
              className={`h-full rounded-full transition-all duration-500 ${
                metrics.cosineSim > 0.6
                  ? 'bg-emerald-500'
                  : metrics.cosineSim > 0.25
                  ? 'bg-amber-500'
                  : 'bg-rose-500'
              }`}
              style={{
                width: `${Math.max(5, Math.min(100, ((metrics.cosineSim + 1) / 2) * 100))}%`,
              }}
            />
          </div>

          <div className="flex justify-between text-[10px] text-slate-400 font-mono">
            <span>-1.0 (Full Semantic Shift)</span>
            <span>0.0</span>
            <span>+1.0 (Identical Meaning)</span>
          </div>

          <div className="text-xs text-slate-600 bg-white p-2.5 rounded-lg border border-slate-200">
            <strong>Interpretation: </strong>
            {metrics.cosineSim > 0.65 ? (
              <span className="text-emerald-700">
                Strong alignment with base prototype. The target maintains its canonical literal semantics.
              </span>
            ) : metrics.cosineSim > 0.25 ? (
              <span className="text-amber-700">
                Moderate displacement. Metaphorical extension or partial domain specialization.
              </span>
            ) : (
              <span className="text-rose-700">
                High displacement from prototype. Strong idiomaticity / non-compositional figurative shift.
              </span>
            )}
          </div>
        </div>

        {/* Displacement Vector Dimensions */}
        <div className="p-4 rounded-xl bg-slate-50 border border-slate-200/80 space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-xs font-bold text-slate-700 uppercase tracking-wider">
              Directional Displacement Δh (Sample Latents)
            </span>
            <span className="text-xs text-slate-500 font-mono">
              d=8 projection
            </span>
          </div>

          <div className="grid grid-cols-4 gap-1.5 pt-1">
            {metrics.displacementVector.map((val, idx) => (
              <div
                key={idx}
                className="bg-white p-1.5 rounded border border-slate-200 text-center font-mono text-[11px]"
              >
                <div className="text-[9px] text-slate-400">dim_{idx}</div>
                <div className={`font-semibold ${val >= 0 ? 'text-blue-700' : 'text-purple-700'}`}>
                  {val >= 0 ? `+${val.toFixed(2)}` : val.toFixed(2)}
                </div>
              </div>
            ))}
          </div>

          <p className="text-[11px] text-slate-500 pt-1">
            Supervised during training with <strong className="text-slate-700 font-mono">prototype_rank_loss</strong>, ensuring pairwise cosine ranking matches human compositionality ordering.
          </p>
        </div>
      </div>
    </div>
  );
};
