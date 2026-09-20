import React, { useState, useMemo } from 'react';
import { BenchmarkExample, TargetType } from './types';
import { BENCHMARK_EXAMPLES } from './data/benchmarkDataset';
import { alignSpansInContext, buildTokenChunks } from './engine/morphology';
import { predictCompositionality, simulateTwoStreamVectors } from './engine/motune';
import { Header } from './components/Header';
import { SentenceInput } from './components/SentenceInput';
import { SpanVisualizer } from './components/SpanVisualizer';
import { DistributionViewer } from './components/DistributionViewer';
import { PrototypeStreamViewer } from './components/PrototypeStreamViewer';
import { LossInspector } from './components/LossInspector';
import { TwoStreamVisualizer } from './components/TwoStreamVisualizer';
import { CurrentModelFlowVisualizer } from './components/CurrentModelFlowVisualizer';
import { DocsModal } from './components/DocsModal';

export const App: React.FC = () => {
  // Default to English "flea market" example
  const defaultExample = BENCHMARK_EXAMPLES[0];

  const [selectedExample, setSelectedExample] = useState<BenchmarkExample | null>(defaultExample);
  const [sentence, setSentence] = useState<string>(defaultExample.sentence);
  const [modWord, setModWord] = useState<string>(defaultExample.mod);
  const [headWord, setHeadWord] = useState<string>(defaultExample.head);
  const [activeTarget, setActiveTarget] = useState<TargetType>(defaultExample.target);
  const [docsOpen, setDocsOpen] = useState<boolean>(false);

  // Synchronize when a benchmark example is selected
  const handleSelectExample = (example: BenchmarkExample) => {
    setSelectedExample(example);
    setSentence(example.sentence);
    setModWord(example.mod);
    setHeadWord(example.head);
    setActiveTarget(example.target);
  };

  const handleResetToDefault = () => {
    handleSelectExample(BENCHMARK_EXAMPLES[0]);
  };

  // Align spans without markers (src/marks.py)
  const spans = useMemo(() => {
    return alignSpansInContext(sentence, modWord, headWord);
  }, [sentence, modWord, headWord]);

  // Token chunks for visual rendering
  const tokens = useMemo(() => {
    return buildTokenChunks(sentence, spans);
  }, [sentence, spans]);

  // Active target word for prototype stream
  const targetWord = useMemo(() => {
    if (activeTarget === 'mod') return modWord;
    if (activeTarget === 'head') return headWord;
    return `${modWord} ${headWord}`.trim();
  }, [activeTarget, modWord, headWord]);

  // Two-stream vectors & cosine metrics (src/model_two_stream.py)
  const prototypeMetrics = useMemo(() => {
    return simulateTwoStreamVectors(targetWord, sentence, activeTarget);
  }, [targetWord, sentence, activeTarget]);

  // Gaussian predictions (mu, sigma, confidence intervals, KL divergence)
  const prediction = useMemo(() => {
    const isMatchingExample =
      selectedExample &&
      selectedExample.sentence === sentence &&
      selectedExample.mod.toLowerCase() === modWord.toLowerCase() &&
      selectedExample.head.toLowerCase() === headWord.toLowerCase() &&
      selectedExample.target === activeTarget;

    const goldMu = isMatchingExample ? selectedExample.goldMu : undefined;
    const goldSigma = isMatchingExample ? selectedExample.goldSigma : undefined;

    return predictCompositionality(
      sentence,
      modWord,
      headWord,
      activeTarget,
      goldMu,
      goldSigma
    );
  }, [sentence, modWord, headWord, activeTarget, selectedExample]);

  return (
    <div className="min-h-screen bg-slate-100/60 text-slate-900 pb-16">
      {/* Header */}
      <Header
        activeTarget={activeTarget}
        onTargetChange={setActiveTarget}
        onOpenDocs={() => setDocsOpen(true)}
      />

      {/* Main Container */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 pt-6 space-y-6">
        {/* Top: Input Form & Benchmark Picker */}
        <SentenceInput
          sentence={sentence}
          onSentenceChange={setSentence}
          modWord={modWord}
          onModChange={setModWord}
          headWord={headWord}
          onHeadChange={setHeadWord}
          activeTarget={activeTarget}
          onTargetChange={setActiveTarget}
          selectedExample={selectedExample}
          onSelectExample={handleSelectExample}
          onResetToDefault={handleResetToDefault}
        />

        {/* Section 1: End-to-End Model Pipeline Flow (Two-Stream Bi-Encoder) */}
        <CurrentModelFlowVisualizer
          sentence={sentence}
          targetWord={targetWord}
          activeTarget={activeTarget}
          spans={spans}
          metrics={prototypeMetrics}
          prediction={prediction}
        />

        {/* Section 2: Marker-Free Span Alignment */}
        <SpanVisualizer
          tokens={tokens}
          spans={spans}
          activeTarget={activeTarget}
        />

        {/* Section 3: Distribution Viewer & Predictions */}
        <DistributionViewer
          prediction={prediction}
          activeTarget={activeTarget}
        />

        {/* Section 4: Two-Stream Prototype & Semantic Displacement */}
        <PrototypeStreamViewer
          metrics={prototypeMetrics}
          targetWord={targetWord}
          activeTarget={activeTarget}
        />

        {/* Section 5: True Two-Stream Bi-Encoder Architecture */}
        <TwoStreamVisualizer
          sentence={sentence}
          targetWord={targetWord}
          activeTarget={activeTarget}
          metrics={prototypeMetrics}
        />

        {/* Section 6: Loss Calibration & Inverted Softplus Bias */}
        <LossInspector />
      </main>

      {/* Full Techniques Documentation Modal */}
      <DocsModal isOpen={docsOpen} onClose={() => setDocsOpen(false)} />
    </div>
  );
};
