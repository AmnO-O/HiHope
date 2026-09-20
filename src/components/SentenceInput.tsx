import React, { useState } from 'react';
import { BenchmarkExample, Language, TargetType } from '../types';
import { BENCHMARK_EXAMPLES } from '../data/benchmarkDataset';
import { Sparkles, Languages, HelpCircle, RotateCcw } from 'lucide-react';

interface SentenceInputProps {
  sentence: string;
  onSentenceChange: (text: string) => void;
  modWord: string;
  onModChange: (mod: string) => void;
  headWord: string;
  onHeadChange: (head: string) => void;
  activeTarget: TargetType;
  onTargetChange: (target: TargetType) => void;
  selectedExample: BenchmarkExample | null;
  onSelectExample: (example: BenchmarkExample) => void;
  onResetToDefault: () => void;
}

export const SentenceInput: React.FC<SentenceInputProps> = ({
  sentence,
  onSentenceChange,
  modWord,
  onModChange,
  headWord,
  onHeadChange,
  activeTarget,
  onTargetChange,
  selectedExample,
  onSelectExample,
  onResetToDefault,
}) => {
  const [filterLang, setFilterLang] = useState<Language | 'all'>('all');

  const filteredExamples = BENCHMARK_EXAMPLES.filter(
    ex => filterLang === 'all' || ex.language === filterLang
  );

  return (
    <div className="bg-white rounded-xl border border-slate-200/80 shadow-xs p-5 space-y-4">
      {/* Benchmark Presets Bar */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 pb-3 border-b border-slate-100">
        <div className="flex items-center space-x-2">
          <Sparkles className="w-4 h-4 text-indigo-600" />
          <span className="text-xs font-bold uppercase tracking-wider text-slate-700">
            Benchmark Presets (NCTTI & SemEval)
          </span>
        </div>

        <div className="flex items-center space-x-2">
          <span className="text-xs text-slate-500 font-medium">Filter:</span>
          <div className="inline-flex rounded-lg border border-slate-200 p-0.5 bg-slate-50 text-xs">
            <button
              onClick={() => setFilterLang('all')}
              className={`px-2 py-0.5 rounded-md font-medium transition-colors ${
                filterLang === 'all' ? 'bg-white text-slate-900 shadow-xs' : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              All ({BENCHMARK_EXAMPLES.length})
            </button>
            <button
              onClick={() => setFilterLang('en')}
              className={`px-2 py-0.5 rounded-md font-medium transition-colors ${
                filterLang === 'en' ? 'bg-white text-slate-900 shadow-xs' : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              English
            </button>
            <button
              onClick={() => setFilterLang('de')}
              className={`px-2 py-0.5 rounded-md font-medium transition-colors ${
                filterLang === 'de' ? 'bg-white text-slate-900 shadow-xs' : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              German
            </button>
          </div>
        </div>
      </div>

      {/* Preset Chips */}
      <div className="flex flex-wrap gap-2 max-h-32 overflow-y-auto pr-1">
        {filteredExamples.map(example => {
          const isSelected = selectedExample?.id === example.id && selectedExample?.target === activeTarget;
          return (
            <button
              key={`${example.id}-${example.target}`}
              id={`preset-${example.id}`}
              onClick={() => {
                onSelectExample(example);
                onTargetChange(example.target);
              }}
              className={`text-xs px-2.5 py-1.5 rounded-lg border transition-all text-left flex items-center space-x-1.5 ${
                isSelected
                  ? 'bg-indigo-50 border-indigo-300 text-indigo-900 font-semibold shadow-xs'
                  : 'bg-slate-50 hover:bg-slate-100 border-slate-200 text-slate-700'
              }`}
            >
              <span className="text-[10px] px-1 py-0.2 rounded bg-slate-200 text-slate-600 uppercase font-mono font-bold">
                {example.language}
              </span>
              <span className="font-medium">{example.expression}</span>
              <span className="text-slate-400 text-[10px]">({example.target})</span>
            </button>
          );
        })}
      </div>

      {/* Interactive Input Form */}
      <div className="space-y-3 pt-2">
        <div>
          <div className="flex items-center justify-between mb-1.5">
            <label htmlFor="context-sentence-input" className="text-xs font-bold text-slate-700 uppercase tracking-wider">
              Context Sentence (Raw text, no artificial markers)
            </label>
            <button
              onClick={onResetToDefault}
              className="text-xs text-slate-500 hover:text-indigo-600 flex items-center space-x-1 transition-colors"
              title="Reset to default example"
            >
              <RotateCcw className="w-3 h-3" />
              <span>Reset</span>
            </button>
          </div>
          <textarea
            id="context-sentence-input"
            value={sentence}
            onChange={e => onSentenceChange(e.target.value)}
            rows={2}
            className="w-full px-3 py-2 text-sm bg-slate-50 border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 text-slate-900 font-medium"
            placeholder="Enter a context sentence containing a compound or particle verb..."
          />
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          <div>
            <label htmlFor="mod-word-input" className="block text-xs font-bold text-blue-700 uppercase tracking-wider mb-1">
              Modifier Lemma / Particle (Mod)
            </label>
            <div className="relative">
              <input
                id="mod-word-input"
                type="text"
                value={modWord}
                onChange={e => onModChange(e.target.value)}
                className="w-full px-3 py-1.5 text-sm bg-blue-50/40 border border-blue-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 text-slate-900 font-medium"
                placeholder="e.g. flea, silver, ab"
              />
              <span className="absolute right-2.5 top-2 text-[11px] text-blue-600/70 font-mono font-semibold">
                mod
              </span>
            </div>
          </div>

          <div>
            <label htmlFor="head-word-input" className="block text-xs font-bold text-purple-700 uppercase tracking-wider mb-1">
              Head Lemma / Verb (Head)
            </label>
            <div className="relative">
              <input
                id="head-word-input"
                type="text"
                value={headWord}
                onChange={e => onHeadChange(e.target.value)}
                className="w-full px-3 py-1.5 text-sm bg-purple-50/40 border border-purple-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-500 text-slate-900 font-medium"
                placeholder="e.g. market, screen, hauen"
              />
              <span className="absolute right-2.5 top-2 text-[11px] text-purple-600/70 font-mono font-semibold">
                head
              </span>
            </div>
          </div>
        </div>

        {selectedExample?.notes && (
          <div className="p-2.5 rounded-lg bg-indigo-50/60 border border-indigo-100 flex items-start space-x-2 text-xs text-indigo-900">
            <HelpCircle className="w-4 h-4 text-indigo-600 shrink-0 mt-0.5" />
            <div>
              <span className="font-semibold">Linguistic Note: </span>
              {selectedExample.notes}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
