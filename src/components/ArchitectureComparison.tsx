import React from 'react';
import { ModelBackend, TargetType } from '../types';
import { Layers, Network, ArrowDown, Sparkles, CheckCircle } from 'lucide-react';

interface ArchitectureComparisonProps {
  backend: ModelBackend;
  activeTarget: TargetType;
  onBackendChange: (b: ModelBackend) => void;
}

export const ArchitectureComparison: React.FC<ArchitectureComparisonProps> = ({
  backend,
  activeTarget,
  onBackendChange,
}) => {
  return (
    <div className="bg-white rounded-xl border border-slate-200/80 shadow-xs p-5 space-y-4">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 pb-3 border-b border-slate-100">
        <div className="flex items-center space-x-2">
          <Layers className="w-4 h-4 text-indigo-600" />
          <h3 className="text-xs font-bold uppercase tracking-wider text-slate-800">
            Model Backend Architecture Comparison
          </h3>
        </div>

        <div className="flex items-center space-x-2">
          <span className="text-xs text-slate-500 font-medium">Selected:</span>
          <span className="px-2.5 py-0.5 rounded-full text-xs font-semibold bg-indigo-50 text-indigo-700 border border-indigo-200">
            {backend === 'exits' ? 'Multi-Exit Backend (src/model.py)' : 'Combined Backend (src/model_combined.py)'}
          </span>
        </div>
      </div>

      {/* 3-way Architecture comparison */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {/* Card 1: Two-Stream Bi-Encoder (User Idea & Recommended) */}
        <div
          onClick={() => onBackendChange('twostream')}
          className={`cursor-pointer p-4 rounded-xl border transition-all text-left space-y-3 ${
            backend === 'twostream'
              ? 'bg-indigo-50/50 border-indigo-400 ring-2 ring-indigo-500/20 shadow-xs'
              : 'bg-slate-50 hover:bg-slate-100/70 border-slate-200'
          }`}
        >
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <Sparkles className={`w-4 h-4 ${backend === 'twostream' ? 'text-indigo-600' : 'text-slate-500'}`} />
              <h4 className="text-sm font-bold text-slate-900">Two-Stream Bi-Encoder</h4>
            </div>
            <span className="text-[10px] font-mono font-semibold px-1.5 py-0.5 rounded bg-emerald-100 text-emerald-800">
              Optimal
            </span>
          </div>

          <p className="text-xs text-slate-600 leading-relaxed">
            Bi-encoder duyệt song song <strong>Target Word cô lập</strong> (h_word) và <strong>Sentence Context</strong> (h_context):
          </p>

          <div className="space-y-2 pt-1">
            <div className={`p-2 rounded-lg text-xs font-mono border flex items-center justify-between ${
              backend === 'twostream' ? 'bg-blue-100/70 border-blue-300 text-blue-950 font-semibold' : 'bg-white border-slate-200 text-slate-700'
            }`}>
              <span>Stream 1: <strong className="font-sans">h_word</strong> (Prototype)</span>
              <span className="text-[10px] px-1.5 py-0.2 rounded bg-slate-100 text-slate-600">Isolated</span>
            </div>

            <div className={`p-2 rounded-lg text-xs font-mono border flex items-center justify-between ${
              backend === 'twostream' ? 'bg-purple-100/70 border-purple-300 text-purple-950 font-semibold' : 'bg-white border-slate-200 text-slate-700'
            }`}>
              <span>Stream 2: <strong className="font-sans">h_context</strong> (In-Context)</span>
              <span className="text-[10px] px-1.5 py-0.2 rounded bg-slate-100 text-slate-600">Contextual</span>
            </div>

            <div className="p-2 rounded-lg text-xs font-mono bg-white border border-slate-200 text-slate-700 flex items-center justify-between">
              <span>Displacement: <strong className="font-sans">Δh + Cos + Prod</strong></span>
              <span className="text-[10px] px-1.5 py-0.2 rounded bg-slate-100 text-slate-600">Fusion</span>
            </div>
          </div>

          <div className="pt-2 text-xs text-slate-600 space-y-1">
            <div className="flex items-center space-x-1.5">
              <CheckCircle className="w-3.5 h-3.5 text-emerald-600 shrink-0" />
              <span><strong>Không cần LoRA</strong>: Full fine-tuning ổn định</span>
            </div>
            <div className="flex items-center space-x-1.5">
              <CheckCircle className="w-3.5 h-3.5 text-emerald-600 shrink-0" />
              <span><strong>Đo độ lệch ngữ nghĩa trực tiếp</strong></span>
            </div>
          </div>
        </div>

        {/* Card 2: Combined Architecture */}
        <div
          onClick={() => onBackendChange('combined')}
          className={`cursor-pointer p-4 rounded-xl border transition-all text-left space-y-3 ${
            backend === 'combined'
              ? 'bg-indigo-50/40 border-indigo-300 ring-2 ring-indigo-500/20 shadow-xs'
              : 'bg-slate-50 hover:bg-slate-100/70 border-slate-200'
          }`}
        >
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <Network className={`w-4 h-4 ${backend === 'combined' ? 'text-indigo-600' : 'text-slate-500'}`} />
              <h4 className="text-sm font-bold text-slate-900">Combined Backend</h4>
            </div>
            <span className="text-[11px] font-mono text-slate-500">model_combined.py</span>
          </div>

          <p className="text-xs text-slate-600 leading-relaxed">
            Routes context qua 22 layers của mmBERT với 1 pass duy nhất:
          </p>

          <div className="space-y-2 pt-1">
            <div className="p-2 rounded-lg text-xs font-mono bg-white border border-slate-200 text-slate-700 flex items-center justify-between">
              <span>Full Encoder Depth: <strong className="font-sans">Layers 1..22</strong></span>
              <span className="text-[10px] px-1.5 py-0.2 rounded bg-slate-100 text-slate-600">mmBERT</span>
            </div>

            <div className={`p-2 rounded-lg text-xs font-mono border flex items-center justify-between ${
              backend === 'combined' ? 'bg-indigo-100/70 border-indigo-300 text-indigo-900 font-semibold' : 'bg-white border-slate-200 text-slate-700'
            }`}>
              <span>Active Target Pooling: <strong className="font-sans">{activeTarget} span</strong></span>
              <span className="text-[10px] px-1.5 py-0.2 rounded bg-slate-100 text-slate-600">pool_active</span>
            </div>

            <div className="p-2 rounded-lg text-xs font-mono bg-white border border-slate-200 text-slate-700 flex items-center justify-between">
              <span>Single Head: <strong className="font-sans">Shared GaussHead</strong></span>
              <span className="text-[10px] px-1.5 py-0.2 rounded bg-slate-100 text-slate-600">μ + σ</span>
            </div>
          </div>

          <div className="pt-2 text-xs text-slate-600 space-y-1">
            <div className="flex items-center space-x-1.5">
              <CheckCircle className="w-3.5 h-3.5 text-indigo-600 shrink-0" />
              <span><strong>3N Data Augmentation</strong>: 1 câu mở rộng thành 3 mục tiêu</span>
            </div>
            <div className="flex items-center space-x-1.5">
              <CheckCircle className="w-3.5 h-3.5 text-indigo-600 shrink-0" />
              <span><strong>Không so sánh prototype</strong> ngoài câu</span>
            </div>
          </div>
        </div>

        {/* Card 3: Multi-Exit Architecture */}
        <div
          onClick={() => onBackendChange('exits')}
          className={`cursor-pointer p-4 rounded-xl border transition-all text-left space-y-3 ${
            backend === 'exits'
              ? 'bg-indigo-50/40 border-indigo-300 ring-2 ring-indigo-500/20 shadow-xs'
              : 'bg-slate-50 hover:bg-slate-100/70 border-slate-200'
          }`}
        >
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <Layers className={`w-4 h-4 ${backend === 'exits' ? 'text-indigo-600' : 'text-slate-500'}`} />
              <h4 className="text-sm font-bold text-slate-900">Multi-Exit Backend</h4>
            </div>
            <span className="text-[11px] font-mono text-slate-500">model.py</span>
          </div>

          <p className="text-xs text-slate-600 leading-relaxed">
            Ngắt sớm representation ở các tầng trung gian (Layer 18, 19, 21):
          </p>

          <div className="space-y-2 pt-1">
            <div className={`p-2 rounded-lg text-xs font-mono border flex items-center justify-between ${
              activeTarget === 'mod' && backend === 'exits' ? 'bg-blue-100/70 border-blue-300 text-blue-900' : 'bg-white border-slate-200 text-slate-700'
            }`}>
              <span>Layer 18: <strong className="font-sans">Modifier Exit</strong></span>
              <span className="text-[10px] px-1.5 py-0.2 rounded bg-slate-100 text-slate-600">gauss_mod</span>
            </div>

            <div className={`p-2 rounded-lg text-xs font-mono border flex items-center justify-between ${
              activeTarget === 'head' && backend === 'exits' ? 'bg-purple-100/70 border-purple-300 text-purple-900' : 'bg-white border-slate-200 text-slate-700'
            }`}>
              <span>Layer 19: <strong className="font-sans">Head Exit</strong></span>
              <span className="text-[10px] px-1.5 py-0.2 rounded bg-slate-100 text-slate-600">gauss_head</span>
            </div>

            <div className={`p-2 rounded-lg text-xs font-mono border flex items-center justify-between ${
              activeTarget === 'pv' && backend === 'exits' ? 'bg-emerald-100/70 border-emerald-300 text-emerald-900' : 'bg-white border-slate-200 text-slate-700'
            }`}>
              <span>Layer 21: <strong className="font-sans">Compound Exit</strong></span>
              <span className="text-[10px] px-1.5 py-0.2 rounded bg-slate-100 text-slate-600">SpanFusion</span>
            </div>
          </div>

          <div className="pt-2 text-xs text-slate-600 space-y-1">
            <div className="flex items-center space-x-1.5">
              <CheckCircle className="w-3.5 h-3.5 text-indigo-600 shrink-0" />
              <span><strong>CrossSpanAttentionBlock</strong> trước khi pool</span>
            </div>
            <div className="flex items-center space-x-1.5">
              <CheckCircle className="w-3.5 h-3.5 text-amber-600 shrink-0" />
              <span><strong>Bỏ lỡ 3-4 tầng trừu tượng cao nhất</strong></span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
