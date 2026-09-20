import React, { useState } from 'react';
import { computeGaussKL, computeInverseSoftplusBias, SIGMA_FLOOR } from '../engine/motune';
import { Calculator, HelpCircle, ArrowRight } from 'lucide-react';

export const LossInspector: React.FC = () => {
  const [muP, setMuP] = useState(2.8);
  const [sigmaP, setSigmaP] = useState(0.45);
  const [muT, setMuT] = useState(3.0);
  const [sigmaT, setSigmaT] = useState(0.5);

  const klValue = computeGaussKL(muP, sigmaP, muT, sigmaT);
  const biasInit = computeInverseSoftplusBias(0.5, SIGMA_FLOOR);

  return (
    <div className="bg-white rounded-xl border border-slate-200/80 shadow-xs p-5 space-y-4">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 pb-3 border-b border-slate-100">
        <div className="flex items-center space-x-2">
          <Calculator className="w-4 h-4 text-indigo-600" />
          <h3 className="text-xs font-bold uppercase tracking-wider text-slate-800">
            Loss Calibration & Softplus Bias (<span className="font-mono text-indigo-600">src/losses.py & src/heads.py</span>)
          </h3>
        </div>

        <span className="text-xs text-slate-500 font-mono">
          sigma_floor = {SIGMA_FLOOR}
        </span>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Interactive Sliders */}
        <div className="space-y-4 bg-slate-50 p-4 rounded-xl border border-slate-200/80">
          <div className="text-xs font-bold uppercase tracking-wider text-slate-700">
            Interactive Distribution Parameters
          </div>

          <div className="space-y-3">
            <div>
              <div className="flex justify-between text-xs font-medium text-slate-700 mb-1">
                <span>Predicted Mean (μ_p):</span>
                <span className="font-mono font-bold text-indigo-600">{muP.toFixed(2)}</span>
              </div>
              <input
                type="range"
                min="0.0"
                max="5.0"
                step="0.05"
                value={muP}
                onChange={e => setMuP(parseFloat(e.target.value))}
                className="w-full accent-indigo-600"
              />
            </div>

            <div>
              <div className="flex justify-between text-xs font-medium text-slate-700 mb-1">
                <span>Predicted Uncertainty (σ_p):</span>
                <span className="font-mono font-bold text-indigo-600">{sigmaP.toFixed(2)}</span>
              </div>
              <input
                type="range"
                min="0.05"
                max="1.50"
                step="0.02"
                value={sigmaP}
                onChange={e => setSigmaP(parseFloat(e.target.value))}
                className="w-full accent-indigo-600"
              />
            </div>

            <div className="border-t border-slate-200 pt-2">
              <div className="flex justify-between text-xs font-medium text-slate-700 mb-1">
                <span>Gold Target Mean (y):</span>
                <span className="font-mono font-bold text-emerald-600">{muT.toFixed(2)}</span>
              </div>
              <input
                type="range"
                min="0.0"
                max="5.0"
                step="0.05"
                value={muT}
                onChange={e => setMuT(parseFloat(e.target.value))}
                className="w-full accent-emerald-600"
              />
            </div>

            <div>
              <div className="flex justify-between text-xs font-medium text-slate-700 mb-1">
                <span>Gold Target Std (σ_t):</span>
                <span className="font-mono font-bold text-emerald-600">{sigmaT.toFixed(2)}</span>
              </div>
              <input
                type="range"
                min="0.05"
                max="1.50"
                step="0.02"
                value={sigmaT}
                onChange={e => setSigmaT(parseFloat(e.target.value))}
                className="w-full accent-emerald-600"
              />
            </div>
          </div>
        </div>

        {/* Real-time Math Output */}
        <div className="space-y-4">
          <div className="p-4 rounded-xl bg-indigo-50/60 border border-indigo-200 space-y-2">
            <div className="text-xs font-bold uppercase tracking-wider text-indigo-900">
              Closed-Form Gaussian KL Divergence
            </div>
            <div className="text-2xl font-bold font-mono text-indigo-950">
              {klValue.toFixed(4)}
            </div>
            <div className="p-2.5 bg-white/90 rounded-lg border border-indigo-100 font-mono text-[11px] text-slate-700 leading-relaxed overflow-x-auto">
              ln(σ_t / σ_p) + (σ_p² + (μ_p - y)²) / (2 · σ_t²) - 0.5
            </div>
            <p className="text-[11px] text-indigo-900/80">
              Jointly trains μ (mean literalness) and σ (human annotator variance) with exact analytic gradients.
            </p>
          </div>

          <div className="p-4 rounded-xl bg-slate-50 border border-slate-200 space-y-2">
            <div className="text-xs font-bold uppercase tracking-wider text-slate-800">
              GaussHead Inverse Softplus Initialization
            </div>
            <div className="flex items-baseline space-x-2">
              <span className="text-xl font-bold font-mono text-slate-900">
                bias_init = {biasInit.toFixed(4)}
              </span>
              <span className="text-xs text-slate-500 font-medium">for σ_init = 0.50</span>
            </div>
            <div className="font-mono text-[11px] text-slate-600 bg-white p-2 rounded border border-slate-200">
              bias = ln(exp(σ_init - floor) - 1)
            </div>
            <p className="text-[11px] text-slate-500">
              Ensures that on epoch 0, softplus(bias) + floor ≈ 0.50, preventing exploding gradients or degenerate predictions before warmup.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};
