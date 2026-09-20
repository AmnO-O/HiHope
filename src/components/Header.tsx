import React from 'react';
import { ModelBackend, TargetType } from '../types';
import { Layers, Network, Activity, Sparkles, BookOpen } from 'lucide-react';

interface HeaderProps {
  backend: ModelBackend;
  onBackendChange: (backend: ModelBackend) => void;
  activeTarget: TargetType;
  onTargetChange: (target: TargetType) => void;
  onOpenDocs: () => void;
}

export const Header: React.FC<HeaderProps> = ({
  backend,
  onBackendChange,
  activeTarget,
  onTargetChange,
  onOpenDocs,
}) => {
  return (
    <header className="bg-white border-b border-slate-200 sticky top-0 z-30 shadow-xs">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-3.5 flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
        <div className="flex items-center space-x-3">
          <div className="w-10 h-10 rounded-xl bg-indigo-600 text-white flex items-center justify-center shadow-sm font-bold text-lg tracking-tight">
            Mo
          </div>
          <div>
            <div className="flex items-center space-x-2">
              <h1 className="text-xl font-bold text-slate-900 tracking-tight">MoTune</h1>
              <span className="px-2 py-0.5 rounded-full text-xs font-semibold bg-indigo-50 text-indigo-700 border border-indigo-200/60">
                Gaussian Compositionality
              </span>
            </div>
            <p className="text-xs text-slate-500 font-medium">
              Multi-exit & Two-stream Prototype Architecture for Noun Compounds & Particle Verbs
            </p>
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-3 w-full md:w-auto justify-between md:justify-end">
          {/* Target Type Selector */}
          <div className="flex items-center bg-slate-100 p-1 rounded-lg border border-slate-200/80 text-xs font-medium">
            <span className="px-2 text-slate-500 text-[11px] uppercase tracking-wider font-semibold">Target:</span>
            <button
              id="target-mod-btn"
              onClick={() => onTargetChange('mod')}
              className={`px-2.5 py-1 rounded-md transition-all ${
                activeTarget === 'mod'
                  ? 'bg-blue-600 text-white font-semibold shadow-xs'
                  : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              Modifier (mod)
            </button>
            <button
              id="target-head-btn"
              onClick={() => onTargetChange('head')}
              className={`px-2.5 py-1 rounded-md transition-all ${
                activeTarget === 'head'
                  ? 'bg-purple-600 text-white font-semibold shadow-xs'
                  : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              Head (head)
            </button>
            <button
              id="target-pv-btn"
              onClick={() => onTargetChange('pv')}
              className={`px-2.5 py-1 rounded-md transition-all ${
                activeTarget === 'pv'
                  ? 'bg-emerald-600 text-white font-semibold shadow-xs'
                  : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              Full Compound / PV
            </button>
          </div>

          {/* Backend Selector */}
          <div className="flex items-center bg-slate-100 p-1 rounded-lg border border-slate-200/80 text-xs font-medium">
            <span className="px-2 text-slate-500 text-[11px] uppercase tracking-wider font-semibold">Backend:</span>
            <button
              id="backend-twostream-btn"
              onClick={() => onBackendChange('twostream')}
              className={`flex items-center space-x-1.5 px-2.5 py-1 rounded-md transition-all ${
                backend === 'twostream'
                  ? 'bg-indigo-600 text-white font-semibold shadow-xs'
                  : 'text-slate-600 hover:text-slate-900'
              }`}
              title="Two-Stream Bi-Encoder (h_word vs h_context semantic displacement)"
            >
              <Sparkles className="w-3.5 h-3.5" />
              <span>Two-Stream</span>
            </button>
            <button
              id="backend-combined-btn"
              onClick={() => onBackendChange('combined')}
              className={`flex items-center space-x-1.5 px-2.5 py-1 rounded-md transition-all ${
                backend === 'combined'
                  ? 'bg-indigo-600 text-white font-semibold shadow-xs'
                  : 'text-slate-600 hover:text-slate-900'
              }`}
              title="Single-pass Layer 22 masked-mean pooling"
            >
              <Network className="w-3.5 h-3.5" />
              <span>Combined</span>
            </button>
            <button
              id="backend-exits-btn"
              onClick={() => onBackendChange('exits')}
              className={`flex items-center space-x-1.5 px-2.5 py-1 rounded-md transition-all ${
                backend === 'exits'
                  ? 'bg-indigo-600 text-white font-semibold shadow-xs'
                  : 'text-slate-600 hover:text-slate-900'
              }`}
              title="Dedicated intermediate transformer exits (layers 18, 19, 21)"
            >
              <Layers className="w-3.5 h-3.5" />
              <span>Multi-Exit</span>
            </button>
          </div>

          <button
            id="open-docs-btn"
            onClick={onOpenDocs}
            className="flex items-center space-x-1.5 px-3 py-1.5 rounded-lg border border-slate-200 bg-white text-slate-700 hover:bg-slate-50 text-xs font-semibold shadow-xs transition-colors"
          >
            <BookOpen className="w-3.5 h-3.5 text-slate-500" />
            <span>Techniques & Docs</span>
          </button>
        </div>
      </div>
    </header>
  );
};
