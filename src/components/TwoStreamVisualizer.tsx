import React from 'react';
import { TargetType, PrototypeMetrics } from '../types';
import { GitCompare, ArrowRight, ArrowDown, Cpu, Sparkles, CheckCircle2, Zap } from 'lucide-react';

interface TwoStreamVisualizerProps {
  sentence: string;
  targetWord: string;
  activeTarget: TargetType;
  metrics: PrototypeMetrics;
}

export const TwoStreamVisualizer: React.FC<TwoStreamVisualizerProps> = ({
  sentence,
  targetWord,
  activeTarget,
  metrics,
}) => {
  return (
    <div className="bg-white rounded-xl border border-slate-200/80 shadow-xs p-5 space-y-5">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 pb-3 border-b border-slate-100">
        <div>
          <div className="flex items-center space-x-2">
            <div className="w-6 h-6 rounded-md bg-indigo-600 text-white flex items-center justify-center">
              <GitCompare className="w-3.5 h-3.5" />
            </div>
            <h3 className="text-sm font-bold text-slate-900 tracking-tight">
              True Two-Stream Bi-Encoder Architecture (<span className="font-mono text-indigo-600">py_src/model_two_stream.py</span>)
            </h3>
          </div>
          <p className="text-xs text-slate-500 mt-0.5">
            Duyệt qua song song <strong>Target Word cô lập</strong> (<span className="font-mono text-indigo-600">h_word</span>) và <strong>Câu Context đầy đủ</strong> (<span className="font-mono text-indigo-600">h_context</span>) để đo độ trôi dạt ngữ nghĩa (semantic displacement)
          </p>
        </div>

        <div className="flex items-center space-x-1.5 text-xs">
          <span className="px-2 py-0.5 rounded-full bg-emerald-50 text-emerald-700 border border-emerald-200 font-medium">
            Direct mmBERT (No LoRA Needed)
          </span>
        </div>
      </div>

      {/* Two-Stream Bi-Encoder Parallel Diagram */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* STREAM 1: Isolated Target Word */}
        <div className="p-4 rounded-xl bg-blue-50/60 border border-blue-200/80 flex flex-col justify-between space-y-3">
          <div className="space-y-1.5">
            <div className="flex items-center justify-between">
              <span className="text-xs font-bold uppercase tracking-wider text-blue-900 flex items-center space-x-1.5">
                <span className="w-5 h-5 rounded-full bg-blue-600 text-white flex items-center justify-center text-[10px]">1</span>
                <span>Stream 1: Isolated Target Word</span>
              </span>
              <span className="text-[10px] font-mono font-semibold px-2 py-0.5 rounded bg-blue-100 text-blue-800">
                Prototype Stream
              </span>
            </div>
            <p className="text-xs text-slate-600">
              Mã hoá từ mục tiêu <strong className="text-blue-900">"{targetWord}"</strong> hoàn toàn độc lập, tách khỏi ngữ cảnh câu để xác lập <em>nghĩa đen cơ bản (literal prototype)</em>.
            </p>
          </div>

          <div className="space-y-2 text-xs font-mono">
            <div className="p-2 bg-white rounded border border-blue-200 text-blue-950 truncate">
              <span className="text-slate-400 text-[10px] block font-sans uppercase">Input Sequence:</span>
              [CLS] {targetWord} [SEP]
            </div>

            <div className="flex items-center justify-center text-blue-400">
              <ArrowDown className="w-4 h-4" />
            </div>

            <div className="p-2 bg-white rounded border border-blue-200">
              <span className="text-slate-400 text-[10px] block font-sans uppercase">Encoder & Pooling:</span>
              mmBERT Layer 22 ➔ <code className="text-blue-700 font-bold">pool_prototype()</code>
              <div className="text-[11px] text-slate-500 font-sans mt-0.5">
                (Loại bỏ [CLS] và [SEP], chỉ pool token từ thực tế)
              </div>
            </div>

            <div className="flex items-center justify-center text-blue-400">
              <ArrowDown className="w-4 h-4" />
            </div>

            <div className="p-2.5 bg-blue-600 text-white rounded shadow-xs text-center">
              <span className="text-[10px] text-blue-200 block uppercase font-sans">Vector Biểu Diễn Nghĩa Gốc:</span>
              <span className="font-bold text-sm">h_word [B, 768]</span>
              <span className="text-[10px] block text-blue-100 font-sans mt-0.5">
                Norm: ||h_word|| = {metrics.protoNorm}
              </span>
            </div>
          </div>
        </div>

        {/* STREAM 2: Sentence Context */}
        <div className="p-4 rounded-xl bg-purple-50/60 border border-purple-200/80 flex flex-col justify-between space-y-3">
          <div className="space-y-1.5">
            <div className="flex items-center justify-between">
              <span className="text-xs font-bold uppercase tracking-wider text-purple-900 flex items-center space-x-1.5">
                <span className="w-5 h-5 rounded-full bg-purple-600 text-white flex items-center justify-center text-[10px]">2</span>
                <span>Stream 2: Full Context Sentence</span>
              </span>
              <span className="text-[10px] font-mono font-semibold px-2 py-0.5 rounded bg-purple-100 text-purple-800">
                In-Context Stream
              </span>
            </div>
            <p className="text-xs text-slate-600">
              Mã hoá toàn bộ câu chứa cụm từ để bắt trọn ngữ cảnh sử dụng thực tế của từ mục tiêu trong câu văn.
            </p>
          </div>

          <div className="space-y-2 text-xs font-mono">
            <div className="p-2 bg-white rounded border border-purple-200 text-purple-950 truncate">
              <span className="text-slate-400 text-[10px] block font-sans uppercase">Input Sequence:</span>
              [CLS] {sentence} [SEP]
            </div>

            <div className="flex items-center justify-center text-purple-400">
              <ArrowDown className="w-4 h-4" />
            </div>

            <div className="p-2 bg-white rounded border border-purple-200">
              <span className="text-slate-400 text-[10px] block font-sans uppercase">Encoder & Span Masking:</span>
              mmBERT Layer 22 ➔ <code className="text-purple-700 font-bold">pool_active_context()</code>
              <div className="text-[11px] text-slate-500 font-sans mt-0.5">
                (Masked-mean pool đúng span vị trí của "{targetWord}" trong câu)
              </div>
            </div>

            <div className="flex items-center justify-center text-purple-400">
              <ArrowDown className="w-4 h-4" />
            </div>

            <div className="p-2.5 bg-purple-600 text-white rounded shadow-xs text-center">
              <span className="text-[10px] text-purple-200 block uppercase font-sans">Vector Biểu Diễn Ngữ Cảnh:</span>
              <span className="font-bold text-sm">h_context [B, 768]</span>
              <span className="text-[10px] block text-purple-100 font-sans mt-0.5">
                Norm: ||h_context|| = {metrics.ctxNorm}
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* INTERACTION & DISPLACEMENT SECTION */}
      <div className="p-4 rounded-xl bg-slate-50 border border-slate-200 space-y-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <Zap className="w-4 h-4 text-amber-500" />
            <h4 className="text-xs font-bold uppercase tracking-wider text-slate-900">
              Tương Tác & Độ Lệch Ngữ Nghĩa (Semantic Interaction & Displacement)
            </h4>
          </div>
          <span className="text-xs font-mono font-semibold px-2 py-0.5 rounded bg-white border border-slate-200 text-slate-700">
            cos(h_ctx, h_word) = {metrics.cosineSim}
          </span>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 text-xs font-mono">
          <div className="p-3 bg-white rounded-lg border border-slate-200 space-y-1">
            <div className="text-[10px] font-sans font-bold text-slate-500 uppercase">1. Vector Lệch Hướng:</div>
            <div className="text-indigo-700 font-bold text-xs">Δh = h_context - h_word</div>
            <div className="text-[11px] text-slate-500 font-sans">
              Độ dài ||Δh|| = <strong className="text-slate-800">{metrics.semanticDisplacementNorm}</strong>. Khi nghĩa bóng (idiom), ||Δh|| lớn!
            </div>
          </div>

          <div className="p-3 bg-white rounded-lg border border-slate-200 space-y-1">
            <div className="text-[10px] font-sans font-bold text-slate-500 uppercase">2. Tương Đồng Cosine:</div>
            <div className="text-indigo-700 font-bold text-xs">cos(h_context, h_word)</div>
            <div className="text-[11px] text-slate-500 font-sans">
              Giá trị = <strong className="text-slate-800">{metrics.cosineSim}</strong> ({metrics.literalnessInference === 'high' ? 'Nghĩa đen nguyên bản' : metrics.literalnessInference === 'idiomatic' ? 'Nghĩa bóng / Ẩn dụ' : 'Chuyển nghĩa trung bình'})
            </div>
          </div>

          <div className="p-3 bg-white rounded-lg border border-slate-200 space-y-1">
            <div className="text-[10px] font-sans font-bold text-slate-500 uppercase">3. Tương Tác Phần Tử:</div>
            <div className="text-indigo-700 font-bold text-xs">h_prod = h_context ⊙ h_word</div>
            <div className="text-[11px] text-slate-500 font-sans">
              Hadamard product giữ lại các chiều ngữ nghĩa giao thoa giữa từ và ngữ cảnh.
            </div>
          </div>
        </div>

        {/* Output prediction block */}
        <div className="p-3 rounded-lg bg-indigo-50/80 border border-indigo-200 flex flex-col sm:flex-row sm:items-center justify-between gap-3 text-xs">
          <div className="flex items-center space-x-2 text-indigo-950 font-sans">
            <CheckCircle2 className="w-4 h-4 text-indigo-600 shrink-0" />
            <span>
              <strong>Fusion & Unified GaussHead:</strong> Ghép chuỗi <code className="font-mono bg-white px-1 py-0.5 rounded border border-indigo-200 text-indigo-900">[h_context; h_word; Δh; h_prod; cos]</code> [B, 3073] ➔ Linear ➔ GaussHead dự đoán chính xác <strong className="text-indigo-700">(μ, σ)</strong> mà không cần LoRA!
            </span>
          </div>
        </div>
      </div>
    </div>
  );
};
