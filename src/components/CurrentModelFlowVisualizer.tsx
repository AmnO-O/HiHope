import React, { useState } from 'react';
import { TargetType, AlignedSpans, PrototypeMetrics, GaussianPrediction } from '../types';
import { 
  GitBranch, 
  Layers, 
  Cpu, 
  ArrowRight, 
  CheckCircle2, 
  Code, 
  Activity, 
  Zap, 
  Sliders, 
  FileText,
  ChevronRight,
  Maximize2
} from 'lucide-react';

interface CurrentModelFlowVisualizerProps {
  sentence: string;
  targetWord: string;
  activeTarget: TargetType;
  spans: AlignedSpans;
  metrics: PrototypeMetrics;
  prediction: GaussianPrediction;
}

export const CurrentModelFlowVisualizer: React.FC<CurrentModelFlowVisualizerProps> = ({
  sentence,
  targetWord,
  activeTarget,
  spans,
  metrics,
  prediction,
}) => {
  const [activeStep, setActiveStep] = useState<number>(0);

  const activeSpan = activeTarget === 'mod' 
    ? spans.modSpan 
    : activeTarget === 'head' 
      ? spans.headSpan 
      : spans.compoundSpan;

  const steps = [
    {
      id: 0,
      title: '1. Input & Alignment',
      subtitle: 'Target Word & Sentence Split',
      tag: 'Raw Text',
      description: 'Nhận diện từ mục tiêu (target) và câu ngữ cảnh (context), xác định vị trí character offsets mà không chèn marker nhân tạo.',
      details: {
        stream1: `Target Word: "${targetWord}" (role: ${activeTarget})`,
        stream2: `Sentence: "${sentence}"`,
        alignment: activeSpan && activeSpan.start !== null 
          ? `Span offset: [${activeSpan.start}, ${activeSpan.end}] in sentence -> "${sentence.slice(activeSpan.start, activeSpan.end ?? undefined)}"`
          : 'Span aligned via character indexing',
        tensor: 'Tokenizers input sequences',
      },
    },
    {
      id: 1,
      title: '2. Dual Tokenization',
      subtitle: 'proto_input_ids & ctx_input_ids',
      tag: 'Tokenizing',
      description: 'Phân tách thành 2 luồng tokens riêng biệt để mmBERT xử lý song song.',
      details: {
        stream1: `Stream 1 Tokens: [CLS] ${targetWord} [SEP] -> shape: [B, L_proto]`,
        stream2: `Stream 2 Tokens: [CLS] ... ${targetWord} ... [SEP] -> shape: [B, L_ctx]`,
        mask: `Span Mask: Vector boolean [B, L_ctx] đánh dấu 1 tại vị trí subwords của "${targetWord}"`,
        tensor: 'proto_input_ids, ctx_input_ids, ctx_attention_mask, target_span_mask',
      },
    },
    {
      id: 2,
      title: '3. Dual-Stream mmBERT',
      subtitle: '22 Transformer Layers',
      tag: 'Backbone',
      description: 'Chạy qua mmBERT-base (22 layers, H=768) với cùng bộ trọng số chia sẻ (Shared Weights), không cần LoRA.',
      details: {
        stream1: `H_proto = mmBERT(proto_input_ids) -> Tensor shape [B, L_proto, 768]`,
        stream2: `H_ctx = mmBERT(ctx_input_ids) -> Tensor shape [B, L_ctx, 768]`,
        note: 'Cả 2 luồng đều nằm trong cùng không gian ẩn tầng 22 (Layer 22 latent space)',
        tensor: 'Hidden states H_proto [B, L_proto, 768], H_ctx [B, L_ctx, 768]',
      },
    },
    {
      id: 3,
      title: '4. Feature Pooling',
      subtitle: 'h_word & h_context',
      tag: 'Vector Pool',
      description: 'Rút trích vector đặc trưng 768-chiều cho nghĩa đen gốc và nghĩa ngữ cảnh thực tế.',
      details: {
        stream1: `h_word = pool_prototype(H_proto) [B, 768] (loại bỏ [CLS]/[SEP], norm = ${metrics.protoNorm})`,
        stream2: `h_context = pool_active_context(H_ctx, span_mask) [B, 768] (masked mean, norm = ${metrics.ctxNorm})`,
        meaning: `h_word đại diện nghĩa gốc; h_context đại diện ngữ nghĩa khi đặt trong câu văn.`,
        tensor: 'h_word [B, 768], h_context [B, 768]',
      },
    },
    {
      id: 4,
      title: '5. Semantic Interaction',
      subtitle: 'Δh, Cosine, & Hadamard',
      tag: 'Displacement',
      description: 'Đo lường độ lệch và sự tương tác giữa nghĩa gốc và nghĩa ngữ cảnh.',
      details: {
        displacement: `Δh = h_context - h_word (Norm = ${metrics.semanticDisplacementNorm} -> đo độ lệch nghĩa)`,
        cosine: `cos(h_context, h_word) = ${metrics.cosineSim} (${metrics.literalnessInference === 'high' ? 'Nghĩa đen' : metrics.literalnessInference === 'idiomatic' ? 'Nghĩa bóng' : 'Chuyển nghĩa'})`,
        hadamard: `h_prod = h_context ⊙ h_word (Tương tác phần tử [B, 768])`,
        tensor: 'Delta_h [B, 768], h_prod [B, 768], cos_sim [B, 1]',
      },
    },
    {
      id: 5,
      title: '6. Fusion & GaussHead',
      subtitle: 'Gaussian Output (μ, σ)',
      tag: 'Prediction',
      description: 'Nối chuỗi tensor, qua mạng MLP tích hợp và GaussHead dự đoán phân phối chuẩn.',
      details: {
        concatenation: `x_fusion = [h_context; h_word; Δh; h_prod; cos] -> shape [B, 3073]`,
        projection: `h_fused = Dropout(GELU(LayerNorm(Linear(3073 -> 768)))) -> shape [B, 768]`,
        head: `GaussHead -> μ = ${prediction.mu} (điểm cấu thành 0-5), σ = ${prediction.sigma} (độ bất đồng ≥ 0.05)`,
        tensor: 'mu [B, 1], sigma [B, 1]',
      },
    },
  ];

  return (
    <div className="bg-white rounded-xl border border-slate-200/80 shadow-xs p-5 space-y-5">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 pb-3 border-b border-slate-100">
        <div>
          <div className="flex items-center space-x-2">
            <div className="w-6 h-6 rounded-md bg-indigo-600 text-white flex items-center justify-center shadow-xs">
              <GitBranch className="w-3.5 h-3.5" />
            </div>
            <h3 className="text-sm font-bold text-slate-900 tracking-tight">
              Interactive Model Pipeline Flow (Current Two-Stream Bi-Encoder)
            </h3>
          </div>
          <p className="text-xs text-slate-500 mt-0.5">
            Dòng dữ liệu chi tiết từ văn bản thô, qua 2 luồng mã hoá mmBERT song song đến đầu ra Gaussian <span className="font-mono text-indigo-600 font-semibold">(μ, σ)</span>
          </p>
        </div>

        <div className="flex items-center space-x-1.5 text-xs">
          <span className="px-2 py-0.5 rounded-md bg-indigo-50 border border-indigo-200/80 text-indigo-700 font-medium">
            Step {activeStep + 1} of {steps.length}
          </span>
        </div>
      </div>

      {/* Horizontal Interactive Step Navigation */}
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-2">
        {steps.map((step) => {
          const isSelected = activeStep === step.id;
          return (
            <button
              key={step.id}
              onClick={() => setActiveStep(step.id)}
              className={`p-2.5 rounded-lg border text-left transition-all relative ${
                isSelected
                  ? 'bg-indigo-50/80 border-indigo-400 ring-2 ring-indigo-500/20 shadow-xs'
                  : 'bg-slate-50 hover:bg-slate-100/80 border-slate-200'
              }`}
            >
              <div className="flex items-center justify-between">
                <span className={`text-[10px] font-bold uppercase tracking-wider ${
                  isSelected ? 'text-indigo-700' : 'text-slate-500'
                }`}>
                  {step.tag}
                </span>
                {isSelected && <span className="w-1.5 h-1.5 rounded-full bg-indigo-600 animate-pulse" />}
              </div>
              <div className={`text-xs font-semibold mt-1 truncate ${
                isSelected ? 'text-indigo-950 font-bold' : 'text-slate-800'
              }`}>
                {step.title}
              </div>
            </button>
          );
        })}
      </div>

      {/* Active Stage Detailed Inspector */}
      <div className="p-4 rounded-xl bg-slate-50/70 border border-slate-200 space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 pb-2 border-b border-slate-200/60">
          <div className="space-y-0.5">
            <div className="flex items-center space-x-2">
              <span className="w-5 h-5 rounded-full bg-indigo-600 text-white flex items-center justify-center text-[10px] font-bold">
                {activeStep + 1}
              </span>
              <h4 className="text-sm font-bold text-slate-900">
                {steps[activeStep].title}: {steps[activeStep].subtitle}
              </h4>
            </div>
            <p className="text-xs text-slate-600 pl-7">
              {steps[activeStep].description}
            </p>
          </div>

          <div className="flex items-center space-x-2 text-xs">
            <button
              disabled={activeStep === 0}
              onClick={() => setActiveStep((prev) => Math.max(0, prev - 1))}
              className="px-2.5 py-1 rounded bg-white border border-slate-200 text-slate-700 disabled:opacity-40 hover:bg-slate-100 text-xs font-medium"
            >
              Previous
            </button>
            <button
              disabled={activeStep === steps.length - 1}
              onClick={() => setActiveStep((prev) => Math.min(steps.length - 1, prev + 1))}
              className="px-2.5 py-1 rounded bg-indigo-600 text-white hover:bg-indigo-700 disabled:opacity-40 text-xs font-semibold shadow-xs"
            >
              Next Step
            </button>
          </div>
        </div>

        {/* Live Step Tensor / Data Details */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-xs font-mono">
          <div className="p-3 bg-white rounded-lg border border-slate-200 space-y-1.5 shadow-2xs">
            <div className="text-[10px] font-sans font-bold uppercase tracking-wider text-blue-700 flex items-center space-x-1.5">
              <span className="w-2 h-2 rounded-full bg-blue-500" />
              <span>Stream 1: Target Word Prototype (h_word)</span>
            </div>
            <div className="text-slate-800 text-xs break-all bg-slate-50 p-2 rounded border border-slate-100">
              {steps[activeStep].details.stream1 || steps[activeStep].details.displacement}
            </div>
            <div className="text-[11px] text-slate-500 font-sans">
              Độc lập khỏi ngữ cảnh, dùng làm mốc so sánh nghĩa đen.
            </div>
          </div>

          <div className="p-3 bg-white rounded-lg border border-slate-200 space-y-1.5 shadow-2xs">
            <div className="text-[10px] font-sans font-bold uppercase tracking-wider text-purple-700 flex items-center space-x-1.5">
              <span className="w-2 h-2 rounded-full bg-purple-500" />
              <span>Stream 2: Full Sentence Context (h_context)</span>
            </div>
            <div className="text-slate-800 text-xs break-all bg-slate-50 p-2 rounded border border-slate-100">
              {steps[activeStep].details.stream2 || steps[activeStep].details.cosine || steps[activeStep].details.projection}
            </div>
            <div className="text-[11px] text-slate-500 font-sans">
              Trích xuất vị trí span của từ khi tương tác với toàn bộ câu văn.
            </div>
          </div>
        </div>

        {/* Summary Tensor Shape Banner */}
        <div className="p-3 bg-indigo-50/70 rounded-lg border border-indigo-200/80 flex flex-col sm:flex-row sm:items-center justify-between gap-2 text-xs font-mono">
          <div className="flex items-center space-x-2 text-indigo-950">
            <Zap className="w-4 h-4 text-indigo-600 shrink-0" />
            <span>
              <strong>Active Tensors:</strong> {steps[activeStep].details.tensor}
            </span>
          </div>
          <div className="text-[11px] font-sans text-indigo-800 font-medium">
            {activeStep === 5 ? '🎯 Output: (μ, σ) hoàn tất' : '⚡ Tự động tính toán theo example đang chọn'}
          </div>
        </div>
      </div>

      {/* Visual Two-Stream Data Flow Diagram */}
      <div className="relative p-4 rounded-xl bg-slate-900 text-slate-100 overflow-x-auto shadow-inner">
        <div className="text-[11px] font-mono text-slate-400 uppercase tracking-wider mb-3 flex items-center space-x-2">
          <Activity className="w-3.5 h-3.5 text-indigo-400" />
          <span>Complete Architectural Pipeline Schematic</span>
        </div>

        <div className="min-w-[650px] flex items-center justify-between text-xs space-x-2 font-mono">
          {/* Node 1: Input */}
          <div className="p-2.5 rounded bg-slate-800 border border-slate-700 text-center w-36 shrink-0">
            <div className="text-[10px] text-slate-400 uppercase">Input</div>
            <div className="text-blue-300 font-bold truncate">"{targetWord}"</div>
            <div className="text-purple-300 truncate text-[10px] mt-0.5">"{sentence.slice(0, 18)}..."</div>
          </div>

          <ArrowRight className="w-4 h-4 text-slate-500 shrink-0" />

          {/* Node 2: mmBERT Encoders */}
          <div className="p-2.5 rounded bg-slate-800 border border-indigo-500/50 text-center w-36 shrink-0">
            <div className="text-[10px] text-indigo-400 uppercase font-bold">Shared mmBERT</div>
            <div className="text-white font-bold">22 Layers</div>
            <div className="text-emerald-400 text-[10px] mt-0.5">No LoRA Needed</div>
          </div>

          <ArrowRight className="w-4 h-4 text-slate-500 shrink-0" />

          {/* Node 3: Pooled Vectors */}
          <div className="p-2.5 rounded bg-slate-800 border border-slate-700 text-center w-36 shrink-0">
            <div className="text-[10px] text-slate-400 uppercase">Feature Pool</div>
            <div className="text-blue-400 font-bold">h_word [768]</div>
            <div className="text-purple-400 font-bold">h_ctx [768]</div>
          </div>

          <ArrowRight className="w-4 h-4 text-slate-500 shrink-0" />

          {/* Node 4: Displacement & Cosine */}
          <div className="p-2.5 rounded bg-slate-800 border border-amber-500/50 text-center w-36 shrink-0">
            <div className="text-[10px] text-amber-400 uppercase font-bold">Interaction</div>
            <div className="text-amber-300 font-bold">Δh = h_ctx - h_w</div>
            <div className="text-slate-300 text-[10px]">cos = {metrics.cosineSim}</div>
          </div>

          <ArrowRight className="w-4 h-4 text-slate-500 shrink-0" />

          {/* Node 5: GaussHead Output */}
          <div className="p-2.5 rounded bg-indigo-950 border border-indigo-400 text-center w-36 shrink-0 shadow-sm">
            <div className="text-[10px] text-indigo-300 uppercase font-bold">GaussHead</div>
            <div className="text-white font-bold text-sm">μ = {prediction.mu}</div>
            <div className="text-indigo-300 text-[10px]">σ = {prediction.sigma}</div>
          </div>
        </div>
      </div>
    </div>
  );
};
