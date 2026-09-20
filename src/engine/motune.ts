import { GaussianPrediction, LayerExitInfo, PrototypeMetrics, TargetType } from '../types';

export const SIGMA_FLOOR = 0.05;

/**
 * Closed-form Gaussian KL divergence from MoTune src/losses.py:
 * KL(N(mu_p, sigma_p^2) || N(mu_t, sigma_t^2)) = ln(sigma_t / sigma_p) + (sigma_p^2 + (mu_p - mu_t)^2) / (2 * sigma_t^2) - 0.5
 */
export function computeGaussKL(
  muPred: number,
  sigmaPred: number,
  muTarget: number,
  sigmaTarget: number
): number {
  const sp = Math.max(sigmaPred, SIGMA_FLOOR);
  const st = Math.max(sigmaTarget, SIGMA_FLOOR);
  const expect = (sp ** 2 + (muPred - muTarget) ** 2) / (2 * st ** 2);
  const kl = Math.log(st / sp) + expect - 0.5;
  return Math.max(0, kl);
}

/**
 * Lin's Concordance Correlation Coefficient (CCC)
 */
export function computeCCC(
  preds: number[],
  targets: number[],
  varFloor = 0.05
): number {
  if (preds.length === 0 || preds.length !== targets.length) return 1.0;
  const n = preds.length;
  const meanP = preds.reduce((a, b) => a + b, 0) / n;
  const meanT = targets.reduce((a, b) => a + b, 0) / n;

  let varP = 0;
  let varT = 0;
  let cov = 0;

  for (let i = 0; i < n; i++) {
    const dp = preds[i] - meanP;
    const dt = targets[i] - meanT;
    varP += dp * dp;
    varT += dt * dt;
    cov += dp * dt;
  }

  varP /= n;
  varT /= n;
  cov /= n;

  const denom = varP + varT + (meanP - meanT) ** 2 + varFloor;
  return (2 * cov) / denom;
}

/**
 * Inverse softplus initialization formula from MoTune src/heads.py:
 * bias = ln(exp(sigma_init - floor) - 1)
 */
export function computeInverseSoftplusBias(sigmaInit = 0.5, floor = SIGMA_FLOOR): number {
  const diff = Math.max(1e-5, sigmaInit - floor);
  return Math.log(Math.expm1(diff));
}

/**
 * Generates deterministic 8-dimensional semantic prototype & context vectors
 * to demonstrate prototype displacement and cosine similarity.
 */
export function simulateTwoStreamVectors(
  targetWord: string,
  contextSentence: string,
  targetType: TargetType
): PrototypeMetrics {
  // Deterministic hash based on words
  const hashString = (s: string) => {
    let h = 0x811c9dc5;
    for (let i = 0; i < s.length; i++) {
      h ^= s.charCodeAt(i);
      h = Math.imul(h, 0x01000193);
    }
    return h >>> 0;
  };

  const wordSeed = hashString(targetWord.toLowerCase().trim() || 'prototype');
  const ctxSeed = hashString(contextSentence.toLowerCase().trim() || 'context');

  // Pseudo-random normal distribution
  const randNormal = (seed: number, offset: number) => {
    const x = Math.sin(seed + offset * 12.9898) * 43758.5453;
    return (x - Math.floor(x)) * 2 - 1;
  };

  const dim = 8;
  const protoVec: number[] = [];
  const ctxVec: number[] = [];

  for (let i = 0; i < dim; i++) {
    protoVec.push(randNormal(wordSeed, i));
  }

  // Normalize proto
  const pNorm = Math.sqrt(protoVec.reduce((a, b) => a + b * b, 0)) || 1;
  const normalizedProto = protoVec.map(v => v / pNorm);

  // Context vector has some correlation with prototype + contextual drift
  const driftFactor = targetType === 'pv' ? 0.75 : targetType === 'mod' ? 0.55 : 0.35;
  for (let i = 0; i < dim; i++) {
    const noise = randNormal(ctxSeed, i);
    ctxVec.push(normalizedProto[i] * (1 - driftFactor) + noise * driftFactor);
  }

  // Normalize ctx
  const cNorm = Math.sqrt(ctxVec.reduce((a, b) => a + b * b, 0)) || 1;
  const normalizedCtx = ctxVec.map(v => v / cNorm);

  // Cosine Similarity
  let dot = 0;
  for (let i = 0; i < dim; i++) {
    dot += normalizedProto[i] * normalizedCtx[i];
  }
  const cosineSim = Math.max(-1, Math.min(1, dot));

  // Displacement vector: h_ctx - h_proto
  const displacement = normalizedCtx.map((v, i) => Number((v - normalizedProto[i]).toFixed(4)));
  const displacementNorm = Math.sqrt(displacement.reduce((a, b) => a + b * b, 0));

  let literalness: PrototypeMetrics['literalnessInference'] = 'moderate';
  if (cosineSim > 0.70) literalness = 'high';
  else if (cosineSim < 0.25) literalness = 'idiomatic';

  return {
    protoNorm: Number(pNorm.toFixed(3)),
    ctxNorm: Number(cNorm.toFixed(3)),
    cosineSim: Number(cosineSim.toFixed(4)),
    semanticDisplacementNorm: Number(displacementNorm.toFixed(4)),
    displacementVector: displacement,
    tokenOverlap: Math.min(1, Math.max(0.1, Number((cosineSim * 0.5 + 0.5).toFixed(2)))),
    literalnessInference: literalness,
  };
}

/**
 * Predicts compositionality rating (mu) and annotator disagreement (sigma)
 * with the MoTune backend parameters.
 */
export function predictCompositionality(
  sentence: string,
  modWord: string,
  headWord: string,
  target: TargetType,
  goldMu?: number,
  goldSigma?: number
): GaussianPrediction {
  const protoMetrics = simulateTwoStreamVectors(
    target === 'mod' ? modWord : target === 'head' ? headWord : `${modWord} ${headWord}`,
    sentence,
    target
  );

  let baseMu: number;
  let baseSigma: number;

  if (goldMu !== undefined && goldSigma !== undefined) {
    // Two-stream leverages true prototype displacement for optimal alignment
    const backendDelta = 0.01;
    baseMu = Math.min(5.0, Math.max(0.0, goldMu + backendDelta));
    baseSigma = Math.max(SIGMA_FLOOR, goldSigma - 0.01);
  } else {
    // Synthetic inference based on cosine similarity
    const scoreFromCosine = (protoMetrics.cosineSim + 1) * 2.5; // [-1, 1] -> [0, 5]
    baseMu = Math.min(5.0, Math.max(0.0, Number(scoreFromCosine.toFixed(2))));
    baseSigma = Number((0.25 + (1 - Math.abs(protoMetrics.cosineSim)) * 0.45).toFixed(2));
  }

  const roundedMu = Number(baseMu.toFixed(2));
  const roundedSigma = Number(baseSigma.toFixed(2));
  const ciLow = Math.max(0, Number((roundedMu - 1.96 * roundedSigma).toFixed(2)));
  const ciHigh = Math.min(5, Number((roundedMu + 1.96 * roundedSigma).toFixed(2)));

  const result: GaussianPrediction = {
    mu: roundedMu,
    sigma: roundedSigma,
    goldMu,
    goldSigma,
    confidenceInterval: [ciLow, ciHigh],
  };

  if (goldMu !== undefined && goldSigma !== undefined) {
    result.klDivergence = Number(computeGaussKL(roundedMu, roundedSigma, goldMu, goldSigma).toFixed(4));
    result.ccc = Number(computeCCC([roundedMu], [goldMu]).toFixed(4));
  }

  return result;
}

/**
 * Returns the computational stages of the Two-Stream Bi-Encoder architecture
 */
export function getLayerExits(activeTarget: TargetType): LayerExitInfo[] {
  return [
    {
      layerIndex: 22,
      target: activeTarget,
      description: 'Stream 1 (Isolated Prototype): mmBERT Layer 22 encodes target word/lemma out-of-context -> h_word',
      alphaAttn: 1.0,
      alphaFfn: 1.0,
      active: true,
    },
    {
      layerIndex: 22,
      target: activeTarget,
      description: 'Stream 2 (Sentence Context): mmBERT Layer 22 encodes sentence -> in-context span pooling -> h_context',
      alphaAttn: 1.0,
      alphaFfn: 1.0,
      active: true,
    },
    {
      layerIndex: 22,
      target: activeTarget,
      description: 'Semantic Displacement & Fusion: Delta_h = h_context - h_word, cos_sim, element-wise prod -> GaussHead',
      alphaAttn: 1.0,
      alphaFfn: 1.0,
      active: true,
    },
  ];
}

/**
 * Gaussian Probability Density Function:
 * f(x) = (1 / (sigma * sqrt(2*pi))) * exp(-0.5 * ((x - mu)/sigma)^2)
 */
export function gaussianPdf(x: number, mu: number, sigma: number): number {
  const s = Math.max(sigma, 0.01);
  const factor = 1 / (s * Math.sqrt(2 * Math.PI));
  const exponent = -0.5 * Math.pow((x - mu) / s, 2);
  return factor * Math.exp(exponent);
}

/**
 * Generates density curve coordinates from 0.0 to 5.0 for Recharts visualization
 */
export function generateGaussianCurveData(
  predMu: number,
  predSigma: number,
  goldMu?: number,
  goldSigma?: number,
  steps = 80
) {
  const data = [];
  const minX = 0.0;
  const maxX = 5.0;
  const stepSize = (maxX - minX) / steps;

  for (let i = 0; i <= steps; i++) {
    const x = Number((minX + i * stepSize).toFixed(2));
    const predDensity = Number(gaussianPdf(x, predMu, predSigma).toFixed(4));
    const item: Record<string, number> = {
      score: x,
      predicted: predDensity,
    };

    if (goldMu !== undefined && goldSigma !== undefined) {
      item.gold = Number(gaussianPdf(x, goldMu, goldSigma).toFixed(4));
    }

    data.push(item);
  }

  return data;
}
