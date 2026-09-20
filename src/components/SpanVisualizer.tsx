import React from 'react';
import { AlignedSpans, TargetType, TokenSpan } from '../types';
import { Crosshair, CheckCircle2, AlertTriangle, Cpu } from 'lucide-react';

interface SpanVisualizerProps {
  tokens: TokenSpan[];
  spans: AlignedSpans;
  activeTarget: TargetType;
}

export const SpanVisualizer: React.FC<SpanVisualizerProps> = ({
  tokens,
  spans,
  activeTarget,
}) => {
  const { modSpan, headSpan, compoundSpan, matchType, fugenDetected, separableVerbFound, degenerate } = spans;

  return (
    <div className="bg-white rounded-xl border border-slate-200/80 shadow-xs p-5 space-y-4">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 pb-3 border-b border-slate-100">
        <div className="flex items-center space-x-2">
          <Crosshair className="w-4 h-4 text-indigo-600" />
          <h3 className="text-xs font-bold uppercase tracking-wider text-slate-800">
            Marker-Free Span Alignment (<span className="font-mono text-indigo-600">src/marks.py</span>)
          </h3>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          {/* Match type badge */}
          <span className="inline-flex items-center px-2 py-0.5 rounded-full text-[11px] font-semibold bg-slate-100 text-slate-700 border border-slate-200">
            Strategy: <strong className="ml-1 uppercase text-slate-900">{matchType}</strong>
          </span>

          {fugenDetected && (
            <span className="inline-flex items-center px-2 py-0.5 rounded-full text-[11px] font-semibold bg-amber-50 text-amber-800 border border-amber-200">
              Fugen morpheme: <strong className="ml-1 font-mono">"{fugenDetected}"</strong>
            </span>
          )}

          {separableVerbFound && (
            <span className="inline-flex items-center px-2 py-0.5 rounded-full text-[11px] font-semibold bg-teal-50 text-teal-800 border border-teal-200">
              German Separable Verb
            </span>
          )}

          {degenerate && (
            <span className="inline-flex items-center px-2 py-0.5 rounded-full text-[11px] font-semibold bg-rose-50 text-rose-800 border border-rose-200">
              <AlertTriangle className="w-3 h-3 mr-1" />
              Degenerate Span
            </span>
          )}
        </div>
      </div>

      {/* Rendered Text with Highlighted Spans */}
      <div className="p-4 rounded-lg bg-slate-50 border border-slate-200/80 text-sm leading-relaxed font-sans text-slate-700">
        {tokens.length === 0 ? (
          <span className="text-slate-400 italic">No sentence entered</span>
        ) : (
          tokens.map((chunk, idx) => {
            if (chunk.role === 'mod') {
              const isActive = activeTarget === 'mod' || activeTarget === 'pv';
              return (
                <span
                  key={idx}
                  className={`inline-block px-1.5 py-0.5 mx-0.5 rounded-md text-xs font-semibold font-mono transition-all ${
                    isActive
                      ? 'bg-blue-600 text-white ring-2 ring-blue-300'
                      : 'bg-blue-100 text-blue-900 border border-blue-300'
                  }`}
                  title={`Modifier span [${chunk.start}, ${chunk.end}]`}
                >
                  {chunk.text}
                  <sub className="ml-1 text-[9px] font-sans opacity-90">MOD</sub>
                </span>
              );
            }

            if (chunk.role === 'fugen') {
              return (
                <span
                  key={idx}
                  className="inline-block px-1 py-0.5 mx-0.2 rounded bg-amber-200 text-amber-900 font-mono text-xs font-bold"
                  title="Fugen linking element"
                >
                  {chunk.text}
                  <sub className="ml-0.5 text-[8px] font-sans">FUGEN</sub>
                </span>
              );
            }

            if (chunk.role === 'head') {
              const isActive = activeTarget === 'head' || activeTarget === 'pv';
              return (
                <span
                  key={idx}
                  className={`inline-block px-1.5 py-0.5 mx-0.5 rounded-md text-xs font-semibold font-mono transition-all ${
                    isActive
                      ? 'bg-purple-600 text-white ring-2 ring-purple-300'
                      : 'bg-purple-100 text-purple-900 border border-purple-300'
                  }`}
                  title={`Head span [${chunk.start}, ${chunk.end}]`}
                >
                  {chunk.text}
                  <sub className="ml-1 text-[9px] font-sans opacity-90">HEAD</sub>
                </span>
              );
            }

            return <span key={idx}>{chunk.text}</span>;
          })
        )}
      </div>

      {/* Offset Mapping Inspection Table */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 text-xs pt-1">
        <div
          className={`p-3 rounded-lg border transition-all ${
            activeTarget === 'mod'
              ? 'bg-blue-50/60 border-blue-300 ring-1 ring-blue-300'
              : 'bg-slate-50 border-slate-200'
          }`}
        >
          <div className="flex items-center justify-between text-blue-900 font-semibold mb-1">
            <span>Modifier Span</span>
            {activeTarget === 'mod' && <span className="text-[10px] bg-blue-600 text-white px-1.5 py-0.2 rounded">Active Target</span>}
          </div>
          <div className="font-mono text-slate-600">
            {modSpan.start !== null ? `Offset: [${modSpan.start}, ${modSpan.end})` : 'Unmatched'}
          </div>
          <p className="text-[11px] text-slate-500 mt-1">
            Pooled in exit 18 (Multi-Exit) or pooled as active slice (Combined)
          </p>
        </div>

        <div
          className={`p-3 rounded-lg border transition-all ${
            activeTarget === 'head'
              ? 'bg-purple-50/60 border-purple-300 ring-1 ring-purple-300'
              : 'bg-slate-50 border-slate-200'
          }`}
        >
          <div className="flex items-center justify-between text-purple-900 font-semibold mb-1">
            <span>Head Span</span>
            {activeTarget === 'head' && <span className="text-[10px] bg-purple-600 text-white px-1.5 py-0.2 rounded">Active Target</span>}
          </div>
          <div className="font-mono text-slate-600">
            {headSpan.start !== null ? `Offset: [${headSpan.start}, ${headSpan.end})` : 'Unmatched'}
          </div>
          <p className="text-[11px] text-slate-500 mt-1">
            Pooled in exit 19 (Multi-Exit) or pooled as active slice (Combined)
          </p>
        </div>

        <div
          className={`p-3 rounded-lg border transition-all ${
            activeTarget === 'pv'
              ? 'bg-emerald-50/60 border-emerald-300 ring-1 ring-emerald-300'
              : 'bg-slate-50 border-slate-200'
          }`}
        >
          <div className="flex items-center justify-between text-emerald-900 font-semibold mb-1">
            <span>Compound Span</span>
            {activeTarget === 'pv' && <span className="text-[10px] bg-emerald-600 text-white px-1.5 py-0.2 rounded">Active Target</span>}
          </div>
          <div className="font-mono text-slate-600">
            {compoundSpan.start !== null ? `Offset: [${compoundSpan.start}, ${compoundSpan.end})` : 'Unmatched'}
          </div>
          <p className="text-[11px] text-slate-500 mt-1">
            Joint span pooled across both constituents (exit 21 or Combined)
          </p>
        </div>
      </div>
    </div>
  );
};
