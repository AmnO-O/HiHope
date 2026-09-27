import React, { useState, useMemo } from 'react';
import { simulateQueryAttention } from '../engine/motune';
import { 
  Network, 
  Layers, 
  Target, 
  Compass, 
  Activity, 
  Sliders, 
  Cpu, 
  HelpCircle,
  Eye,
  CheckCircle,
  BarChart3,
  Share2
} from 'lucide-react';

interface QueryAttentionVisualizerProps {
  sentence: string;
  modWord: string;
  headWord: string;
}

export const QueryAttentionVisualizer: React.FC<QueryAttentionVisualizerProps> = ({
  sentence,
  modWord,
  headWord,
}) => {
  const [selectedSlotIndex, setSelectedSlotIndex] = useState<number>(0);
  const [activeTab, setActiveTab] = useState<'architecture' | 'attention_map' | 'interaction' | 'gauss_heads'>('architecture');

  const simulation = useMemo(() => {
    return simulateQueryAttention(sentence, modWord, headWord);
  }, [sentence, modWord, headWord]);

  const selectedSlot = simulation.slots[selectedSlotIndex];

  return (
    <div className="bg-white rounded-xl shadow-sm border border-slate-200 overflow-hidden">
      {/* Header */}
      <div className="p-5 bg-gradient-to-r from-indigo-900 via-slate-900 to-slate-900 text-white flex flex-wrap items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2">
            <span className="px-2.5 py-0.5 rounded-full text-xs font-semibold bg-indigo-500/30 text-indigo-200 border border-indigo-400/40">
              End-to-End Pipeline
            </span>
            <span className="text-xs text-slate-400">Score Range: 1.0 → 5.0</span>
          </div>
          <h2 className="text-xl font-bold mt-1 text-white flex items-center gap-2">
            <Network className="w-6 h-6 text-indigo-400" />
            Target-Aware Query Cross-Attention & Component Interaction
          </h2>
          <p className="text-xs text-slate-300 mt-0.5">
            Learned slot queries [q_Mod, q_Head, q_Compound] with Explicit Role Injection & Shared GaussHead
          </p>
        </div>

        {/* View Switcher */}
        <div className="flex bg-slate-800/80 p-1 rounded-lg border border-slate-700 text-xs font-medium">
          <button
            onClick={() => setActiveTab('architecture')}
            className={`px-3 py-1.5 rounded-md transition-colors ${
              activeTab === 'architecture' ? 'bg-indigo-600 text-white shadow' : 'text-slate-300 hover:text-white'
            }`}
          >
            Pipeline Overview
          </button>
          <button
            onClick={() => setActiveTab('attention_map')}
            className={`px-3 py-1.5 rounded-md transition-colors ${
              activeTab === 'attention_map' ? 'bg-indigo-600 text-white shadow' : 'text-slate-300 hover:text-white'
            }`}
          >
            Attention Heatmap A[3, S]
          </button>
          <button
            onClick={() => setActiveTab('interaction')}
            className={`px-3 py-1.5 rounded-md transition-colors ${
              activeTab === 'interaction' ? 'bg-indigo-600 text-white shadow' : 'text-slate-300 hover:text-white'
            }`}
          >
            Component Matrix (MHSA)
          </button>
          <button
            onClick={() => setActiveTab('gauss_heads')}
            className={`px-3 py-1.5 rounded-md transition-colors ${
              activeTab === 'gauss_heads' ? 'bg-indigo-600 text-white shadow' : 'text-slate-300 hover:text-white'
            }`}
          >
            GaussHead (μ, σ)
          </button>
        </div>
      </div>

      {/* Main View Area */}
      <div className="p-6">
        {activeTab === 'architecture' && (
          <div className="space-y-6">
            {/* Pipeline Step Grid */}
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
              {/* Layer 0 */}
              <div className="p-4 rounded-xl border border-blue-200 bg-blue-50/40 relative">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-xs font-bold text-blue-700 uppercase tracking-wider">Layer 0</span>
                  <span className="text-[10px] px-2 py-0.5 rounded bg-blue-100 text-blue-800 font-mono">B, S, 768</span>
                </div>
                <h3 className="font-semibold text-slate-800 text-sm flex items-center gap-1.5">
                  <Cpu className="w-4 h-4 text-blue-600" />
                  Encoder & Role Injection
                </h3>
                <p className="text-xs text-slate-600 mt-1">
                  mmBERT trích xuất <code className="bg-blue-100 text-blue-900 px-1 py-0.5 rounded text-[11px]">H_mmBERT</code> kết hợp Role Embedding <code className="bg-blue-100 text-blue-900 px-1 py-0.5 rounded text-[11px]">E_role</code> (0: Context, 1: Mod, 2: Head).
                </p>
                <div className="mt-3 text-[11px] font-mono bg-white p-2 rounded border border-blue-200 text-slate-700">
                  H_final = H_mmBERT + E_role
                </div>
              </div>

              {/* Layer 1 */}
              <div className="p-4 rounded-xl border border-indigo-200 bg-indigo-50/40 relative">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-xs font-bold text-indigo-700 uppercase tracking-wider">Layer 1</span>
                  <span className="text-[10px] px-2 py-0.5 rounded bg-indigo-100 text-indigo-800 font-mono">B, 3, 768</span>
                </div>
                <h3 className="font-semibold text-slate-800 text-sm flex items-center gap-1.5">
                  <Target className="w-4 h-4 text-indigo-600" />
                  Target-Aware MHCA
                </h3>
                <p className="text-xs text-slate-600 mt-1">
                  3 Learned Queries <code className="bg-indigo-100 text-indigo-900 px-1 py-0.5 rounded text-[11px]">Q_base</code> [Mod, Head, Comp] cross-attend vào toàn bộ câu.
                </p>
                <div className="mt-3 text-[11px] font-mono bg-white p-2 rounded border border-indigo-200 text-slate-700">
                  Z = LayerNorm(Q + MHCA(Q, K, V))
                </div>
              </div>

              {/* Layer 2 */}
              <div className="p-4 rounded-xl border border-purple-200 bg-purple-50/40 relative">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-xs font-bold text-purple-700 uppercase tracking-wider">Layer 2</span>
                  <span className="text-[10px] px-2 py-0.5 rounded bg-purple-100 text-purple-800 font-mono">B, 3, 768</span>
                </div>
                <h3 className="font-semibold text-slate-800 text-sm flex items-center gap-1.5">
                  <Share2 className="w-4 h-4 text-purple-600" />
                  Component MHSA
                </h3>
                <p className="text-xs text-slate-600 mt-1">
                  Self-Attention giữa Mod ↔ Head ↔ Compound để tổng hợp thông tin cấu trúc thành tố.
                </p>
                <div className="mt-3 text-[11px] font-mono bg-white p-2 rounded border border-purple-200 text-slate-700">
                  Z' = LayerNorm(Z + MHSA(Z))
                </div>
              </div>

              {/* Layer 3 */}
              <div className="p-4 rounded-xl border border-emerald-200 bg-emerald-50/40 relative">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-xs font-bold text-emerald-700 uppercase tracking-wider">Layer 3</span>
                  <span className="text-[10px] px-2 py-0.5 rounded bg-emerald-100 text-emerald-800 font-mono">(μ, σ)</span>
                </div>
                <h3 className="font-semibold text-slate-800 text-sm flex items-center gap-1.5">
                  <Activity className="w-4 h-4 text-emerald-600" />
                  Shared GaussHead
                </h3>
                <p className="text-xs text-slate-600 mt-1">
                  Dự đoán phân phối Gaussian liên tục: μ ∈ [0, 1] → Score = μ * 5.0, tối ưu qua CCC Loss.
                </p>
                <div className="mt-3 text-[11px] font-mono bg-white p-2 rounded border border-emerald-200 text-slate-700">
                  Score = μ * 5.0, σ &gt; 0.04
                </div>
              </div>
            </div>

            {/* Layer 0 Tokens & Role IDs Visualization */}
            <div className="bg-slate-50 p-4 rounded-xl border border-slate-200">
              <h4 className="text-xs font-bold text-slate-700 uppercase tracking-wider mb-2 flex items-center gap-1.5">
                <Layers className="w-4 h-4 text-slate-500" />
                Layer 0: Input Sequence Tokens & Explicit Role IDs (0: Context, 1: Mod, 2: Head)
              </h4>
              <div className="flex flex-wrap gap-2 mt-3">
                {simulation.tokens.map((t, idx) => {
                  let badgeColor = 'bg-slate-200 text-slate-700 border-slate-300';
                  if (t.roleId === 1) badgeColor = 'bg-blue-100 text-blue-800 border-blue-400 font-semibold ring-2 ring-blue-300/50';
                  if (t.roleId === 2) badgeColor = 'bg-emerald-100 text-emerald-800 border-emerald-400 font-semibold ring-2 ring-emerald-300/50';

                  return (
                    <div
                      key={idx}
                      className={`px-3 py-2 rounded-lg border text-xs flex flex-col items-center gap-1 ${badgeColor}`}
                    >
                      <span className="font-mono text-sm">{t.token}</span>
                      <span className="text-[10px] opacity-75 font-mono">Role {t.roleId}</span>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
        )}

        {activeTab === 'attention_map' && (
          <div className="space-y-5">
            {/* Slot Selector */}
            <div className="flex items-center gap-3">
              <span className="text-xs font-semibold text-slate-600">Active Probing Slot Query:</span>
              <div className="flex gap-2">
                {simulation.slots.map((slot, idx) => (
                  <button
                    key={slot.name}
                    onClick={() => setSelectedSlotIndex(idx)}
                    className={`px-3 py-1.5 rounded-lg text-xs font-medium border transition-all flex items-center gap-1.5 ${
                      selectedSlotIndex === idx
                        ? 'bg-indigo-600 text-white border-indigo-600 shadow-sm'
                        : 'bg-slate-50 text-slate-700 border-slate-300 hover:bg-slate-100'
                    }`}
                  >
                    <Target className="w-3.5 h-3.5" />
                    Slot {idx + 1}: {slot.name} Query
                  </button>
                ))}
              </div>
            </div>

            {/* Attention Heatmap Visualization */}
            <div className="bg-slate-50 p-5 rounded-xl border border-slate-200">
              <div className="flex items-center justify-between mb-3">
                <h4 className="text-xs font-bold text-slate-700 uppercase tracking-wider">
                  Cross-Attention Map A[{selectedSlot.name}, S] over Sentence Tokens
                </h4>
                <span className="text-xs text-slate-500 font-mono">Softmax(Q · Kᵀ / √d_h)</span>
              </div>

              <div className="flex flex-wrap gap-2.5 items-end">
                {selectedSlot.attentionScores.map((score, idx) => {
                  const intensity = Math.min(1.0, score.weight * 3.5);
                  const bgStyle = {
                    backgroundColor: `rgba(79, 70, 229, ${Math.max(0.08, intensity)})`,
                  };

                  return (
                    <div
                      key={idx}
                      style={bgStyle}
                      className={`p-3 rounded-lg border flex flex-col items-center min-w-[70px] transition-all ${
                        score.isTarget ? 'border-indigo-400 shadow-sm ring-1 ring-indigo-300' : 'border-slate-200'
                      }`}
                    >
                      <span className="text-xs font-bold text-slate-900 font-mono">{score.token}</span>
                      <span className="text-[11px] font-mono text-indigo-900 mt-1 font-semibold">
                        {(score.weight * 100).toFixed(1)}%
                      </span>
                      <span className="text-[9px] text-slate-500 uppercase mt-0.5">
                        {score.roleId === 1 ? 'Mod' : score.roleId === 2 ? 'Head' : 'Context'}
                      </span>
                    </div>
                  );
                })}
              </div>

              <div className="mt-4 pt-3 border-t border-slate-200 text-xs text-slate-600 flex items-center justify-between">
                <span>
                  <strong>Insight:</strong> Query Slot <code className="text-indigo-600 font-bold">{selectedSlot.name}</code> chú ý cao vào các từ ngữ cảnh liên quan trực tiếp đến tính thành tố.
                </span>
                <span className="text-slate-500 font-mono">Z ∈ ℝ[1, 768]</span>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'interaction' && (
          <div className="space-y-5">
            <div className="bg-slate-50 p-5 rounded-xl border border-slate-200">
              <h4 className="text-xs font-bold text-slate-700 uppercase tracking-wider mb-2">
                Layer 2: Component Interaction Self-Attention Matrix (3 × 3)
              </h4>
              <p className="text-xs text-slate-600 mb-4">
                Cho phép Mod ↔ Head ↔ Compound tương tác chéo, giúp nhận diện sự biến đổi ngữ nghĩa cụm.
              </p>

              <div className="overflow-x-auto">
                <table className="min-w-full text-xs text-slate-700 border border-slate-300 text-center font-mono">
                  <thead className="bg-slate-200 text-slate-800 font-semibold">
                    <tr>
                      <th className="p-2.5 border border-slate-300 text-left">Query \ Key</th>
                      <th className="p-2.5 border border-slate-300">Mod (z_Mod)</th>
                      <th className="p-2.5 border border-slate-300">Head (z_Head)</th>
                      <th className="p-2.5 border border-slate-300">Compound (z_Comp)</th>
                    </tr>
                  </thead>
                  <tbody>
                    {['Mod (z_Mod)', 'Head (z_Head)', 'Compound (z_Comp)'].map((rowName, rIdx) => (
                      <tr key={rowName} className="hover:bg-indigo-50/50">
                        <td className="p-2.5 font-bold border border-slate-300 text-left bg-slate-100">{rowName}</td>
                        {simulation.componentMatrix[rIdx].map((val, cIdx) => (
                          <td
                            key={cIdx}
                            className="p-2.5 border border-slate-300 font-semibold"
                            style={{
                              backgroundColor: `rgba(147, 51, 234, ${val * 0.4})`,
                            }}
                          >
                            {(val * 100).toFixed(1)}%
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              <div className="mt-4 p-3 bg-purple-50 rounded-lg border border-purple-200 text-xs text-purple-900">
                <strong>Ý nghĩa ngôn ngữ học:</strong> Vector <code>z'_Head</code> sau bước này đã tích hợp đầy đủ mối liên hệ cấu trúc với <code>z_Mod</code>, giải quyết triệt để vấn đề mất thông tin khi pooling độc lập.
              </div>
            </div>
          </div>
        )}

        {activeTab === 'gauss_heads' && (
          <div className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              {simulation.slots.map((slot) => (
                <div key={slot.name} className="p-4 rounded-xl border border-slate-200 bg-slate-50">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-xs font-bold text-slate-700 uppercase tracking-wider">{slot.name} Target</span>
                    <span className="text-xs px-2 py-0.5 rounded bg-emerald-100 text-emerald-800 font-bold">
                      Score: {slot.score.toFixed(2)} / 5.0
                    </span>
                  </div>

                  <div className="mt-3 space-y-2 text-xs font-mono">
                    <div className="flex justify-between p-2 bg-white rounded border border-slate-200">
                      <span className="text-slate-500">Predicted Mean (μ):</span>
                      <span className="font-bold text-indigo-700">{slot.mu.toFixed(3)}</span>
                    </div>
                    <div className="flex justify-between p-2 bg-white rounded border border-slate-200">
                      <span className="text-slate-500">Uncertainty (σ):</span>
                      <span className="font-bold text-emerald-700">{slot.sigma.toFixed(3)}</span>
                    </div>
                    <div className="flex justify-between p-2 bg-white rounded border border-slate-200">
                      <span className="text-slate-500">Continuous Output:</span>
                      <span className="font-bold text-slate-800">μ × 5.0 = {slot.score.toFixed(2)}</span>
                    </div>
                  </div>
                </div>
              ))}
            </div>

            <div className="p-4 bg-emerald-50 rounded-xl border border-emerald-200 text-xs text-emerald-900 flex items-center justify-between">
              <div>
                <strong>Lin's CCC Loss Optimization:</strong> L_CCC(μ_head, y_head_norm) + L_CCC(μ_mod, y_mod_norm)
                <div className="text-[11px] text-emerald-700 mt-0.5">Tối ưu trực tiếp độ tương quan xếp hạng và độ tiệm cận thực tế trên dải liên tục [1.0, 5.0].</div>
              </div>
              <span className="text-sm font-bold font-mono px-3 py-1 bg-white rounded border border-emerald-300 text-emerald-800">
                Loss = {simulation.overallLoss.toFixed(4)}
              </span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
