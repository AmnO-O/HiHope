export type TargetType = 'mod' | 'head' | 'pv';
export type ModelBackend = 'twostream' | 'combined' | 'exits';
export type Language = 'en' | 'de';

export interface Span {
  start: number | null;
  end: number | null;
}

export interface AlignedSpans {
  modSpan: Span;
  headSpan: Span;
  compoundSpan: Span;
  degenerate: boolean;
  matchType: 'fused' | 'spaced' | 'independent' | 'none';
  fugenDetected?: string | null;
  separableVerbFound?: boolean;
}

export interface TokenSpan {
  text: string;
  start: number;
  end: number;
  role: 'none' | 'mod' | 'head' | 'pv' | 'fugen';
}

export interface GaussianPrediction {
  mu: number; // 0.0 to 5.0
  sigma: number; // >= 0.05
  goldMu?: number;
  goldSigma?: number;
  klDivergence?: number;
  ccc?: number;
  confidenceInterval: [number, number]; // 95% CI: [mu - 1.96*sigma, mu + 1.96*sigma]
}

export interface PrototypeMetrics {
  protoNorm: number;
  ctxNorm: number;
  cosineSim: number;
  semanticDisplacementNorm: number;
  displacementVector: number[];
  tokenOverlap: number;
  literalnessInference: 'high' | 'moderate' | 'idiomatic';
}

export interface LayerExitInfo {
  layerIndex: number;
  target: TargetType;
  description: string;
  alphaAttn: number;
  alphaFfn: number;
  active: boolean;
}

export interface BenchmarkExample {
  id: string;
  language: Language;
  expression: string;
  mod: string;
  head: string;
  sentence: string;
  type: 'Noun Compound' | 'Particle Verb';
  target: TargetType;
  goldMu: number;
  goldSigma: number;
  notes: string;
}
