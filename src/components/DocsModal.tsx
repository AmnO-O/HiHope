import React from 'react';
import { X, BookOpen, Check, Layers, GitCompare, Cpu, ShieldCheck } from 'lucide-react';

interface DocsModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export const DocsModal: React.FC<DocsModalProps> = ({ isOpen, onClose }) => {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 bg-slate-900/60 backdrop-blur-xs flex items-center justify-center p-4 overflow-y-auto">
      <div className="bg-white rounded-2xl max-w-4xl w-full max-h-[90vh] flex flex-col shadow-2xl border border-slate-200">
        {/* Modal Header */}
        <div className="px-6 py-4 border-b border-slate-200 flex items-center justify-between sticky top-0 bg-white rounded-t-2xl z-10">
          <div className="flex items-center space-x-2.5">
            <div className="w-8 h-8 rounded-lg bg-indigo-600 text-white flex items-center justify-center">
              <BookOpen className="w-4 h-4" />
            </div>
            <div>
              <h2 className="text-base font-bold text-slate-900">MoTune Framework Architecture & Techniques</h2>
              <p className="text-xs text-slate-500">Comprehensive breakdown of all modules in <span className="font-mono">src/</span></p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 rounded-lg text-slate-400 hover:text-slate-700 hover:bg-slate-100 transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Modal Content */}
        <div className="p-6 overflow-y-auto space-y-6 text-sm text-slate-700 leading-relaxed">
          {/* Section 1 */}
          <div className="space-y-2 border-b border-slate-100 pb-5">
            <h3 className="text-sm font-bold text-slate-900 flex items-center space-x-2">
              <span className="w-5 h-5 rounded-full bg-indigo-100 text-indigo-700 flex items-center justify-center text-xs font-mono font-bold">1</span>
              <span>Marker-Free Span Alignment (<code className="text-indigo-600 font-mono">src/marks.py</code>)</span>
            </h3>
            <p className="text-xs text-slate-600">
              Unlike traditional setups that inject artificial delimiters (such as <code className="bg-slate-100 px-1 py-0.5 rounded">&lt;mod&gt;</code> or <code className="bg-slate-100 px-1 py-0.5 rounded">&lt;head&gt;</code>) into sentences—distorting pretrained positional embeddings—MoTune tokenizes raw text untouched and relies on tokenizer <code className="bg-slate-100 px-1 py-0.5 rounded">offset_mapping</code> to map surface forms back to token spans.
            </p>
            <ul className="text-xs list-disc list-inside space-y-1 text-slate-600 pl-2">
              <li><strong>Fused German Compounds:</strong> Identifies closed compounds (e.g., <em>Abiturzeugnis</em>) where the modifier transitions directly into the head.</li>
              <li><strong>Fugen Element Detection:</strong> Discovers inter-word connecting morphemes (<code className="font-mono">-s-</code>, <code className="font-mono">-en-</code>, <code className="font-mono">-er-</code>, <code className="font-mono">-ens-</code>).</li>
              <li><strong>German Separable Verbs:</strong> Resolves discontinuous verb-particle constructions (e.g. <em>hauen ... ab</em>) with an irregular verb dictionary (<code className="font-mono">_IRREGULAR_DE</code>) capturing ablaut stem alternations.</li>
              <li><strong>Degenerate Span Filtering:</strong> Safely excludes samples where modifier and head collapse onto the identical subword token.</li>
            </ul>
          </div>

          {/* Section 2 */}
          <div className="space-y-2 border-b border-slate-100 pb-5">
            <h3 className="text-sm font-bold text-slate-900 flex items-center space-x-2">
              <span className="w-5 h-5 rounded-full bg-indigo-100 text-indigo-700 flex items-center justify-center text-xs font-mono font-bold">2</span>
              <span>Multi-Exit vs. Combined Backend (<code className="text-indigo-600 font-mono">src/model.py & src/model_combined.py</code>)</span>
            </h3>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 pt-1 text-xs">
              <div className="p-3 rounded-lg bg-slate-50 border border-slate-200">
                <strong className="text-slate-900 block mb-1">Multi-Exit (<code className="font-mono text-indigo-600">src/model.py</code>)</strong>
                Taps representations from intermediate transformer blocks corresponding to conceptual depth:
                <ul className="list-disc list-inside mt-1 space-y-0.5 text-slate-600">
                  <li>Modifier: Layer 18</li>
                  <li>Head: Layer 19</li>
                  <li>Compound / PV: Layers 20–21</li>
                </ul>
                Employs <em>CrossSpanAttentionBlock</em> for pre-pooling token-level constituent interaction, and gated residuals initialized to 1.0 (α_attn=1.0, α_ffn=1.0).
              </div>
              <div className="p-3 rounded-lg bg-slate-50 border border-slate-200">
                <strong className="text-slate-900 block mb-1">Combined (<code className="font-mono text-indigo-600">src/model_combined.py</code>)</strong>
                Routes through all 22 layers of mmBERT, extracts active target spans via <code className="font-mono">pool_active</code>, and shares a single unified <code className="font-mono">GaussHead</code> across all 3 targets, maximizing parameter efficiency on small datasets.
              </div>
            </div>
          </div>

          {/* Section 3 */}
          <div className="space-y-2 border-b border-slate-100 pb-5">
            <h3 className="text-sm font-bold text-slate-900 flex items-center space-x-2">
              <span className="w-5 h-5 rounded-full bg-indigo-100 text-indigo-700 flex items-center justify-center text-xs font-mono font-bold">3</span>
              <span>Two-Stream Prototype Displacement (<code className="text-indigo-600 font-mono">src/prototype_stream.py</code>)</span>
            </h3>
            <p className="text-xs text-slate-600">
              A bi-encoder mechanism comparing isolated out-of-context prototype representation <code className="font-mono">h_proto</code> against in-context representation <code className="font-mono">h_ctx</code>.
            </p>
            <div className="p-3 bg-slate-50 rounded-lg border border-slate-200 text-xs space-y-1 font-mono">
              <div>Cosine Similarity: cos(h_ctx, h_proto) ∈ [-1.0, 1.0]</div>
              <div>Directional Displacement: Δh = h_ctx - h_proto</div>
              <div>Semantic Shift Fusion: Cross-attention over 3-token sequence [h_ctx, h_proto, Δh]</div>
            </div>
          </div>

          {/* Section 4 */}
          <div className="space-y-2 border-b border-slate-100 pb-5">
            <h3 className="text-sm font-bold text-slate-900 flex items-center space-x-2">
              <span className="w-5 h-5 rounded-full bg-indigo-100 text-indigo-700 flex items-center justify-center text-xs font-mono font-bold">4</span>
              <span>Gaussian Head & Inverted Softplus Initialization (<code className="text-indigo-600 font-mono">src/heads.py</code>)</span>
            </h3>
            <p className="text-xs text-slate-600">
              Predicts both central compositionality rating <code className="font-mono">μ</code> and annotator disagreement variance <code className="font-mono">σ ≥ 0.05</code>. To avoid gradient explosion on step 0:
            </p>
            <div className="p-2.5 bg-indigo-50/70 rounded-lg border border-indigo-200 text-xs font-mono text-indigo-950">
              bias_init = ln(exp(σ_init - floor) - 1)
            </div>
            <p className="text-xs text-slate-600">
              Ensures <code className="font-mono">softplus(bias) + floor ≈ 0.50</code> at initialization.
            </p>
          </div>

          {/* Section 5 */}
          <div className="space-y-2">
            <h3 className="text-sm font-bold text-slate-900 flex items-center space-x-2">
              <span className="w-5 h-5 rounded-full bg-indigo-100 text-indigo-700 flex items-center justify-center text-xs font-mono font-bold">5</span>
              <span>True Two-Stream Bi-Encoder & Elimination of LoRA (<code className="text-indigo-600 font-mono">py_src/model_two_stream.py</code>)</span>
            </h3>
            <p className="text-xs text-slate-600">
              Với mô hình mmBERT-base (~110M tham số), việc can thiệp LoRA vào 4 layer trên cùng làm bóp nghẽn gradient và giới hạn năng lực thích ứng đa ngôn ngữ. Kiến trúc Two-Stream Bi-Encoder mới loại bỏ LoRA, huấn luyện trực tiếp qua 2 luồng:
            </p>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 text-xs pt-1">
              <div className="p-2 rounded bg-slate-50 border border-slate-200">
                <strong>Luồng 1 (Target Word):</strong> Mã hoá từ mục tiêu cô lập để lấy điểm tựa nghĩa gốc <code className="font-mono text-indigo-700">h_word</code>.
              </div>
              <div className="p-2 rounded bg-slate-50 border border-slate-200">
                <strong>Luồng 2 (Sentence Context):</strong> Mã hoá câu và đo trực tiếp vector độ lệch <code className="font-mono text-indigo-700">Δh = h_context - h_word</code>.
              </div>
            </div>
          </div>
        </div>

        {/* Modal Footer */}
        <div className="px-6 py-3.5 border-t border-slate-200 bg-slate-50 rounded-b-2xl flex justify-end">
          <button
            onClick={onClose}
            className="px-4 py-2 bg-indigo-600 hover:bg-indigo-700 text-white text-xs font-semibold rounded-lg shadow-xs transition-colors"
          >
            Close Reference
          </button>
        </div>
      </div>
    </div>
  );
};
