import React from 'react';
import { GaussianPrediction, TargetType } from '../types';
import { generateGaussianCurveData } from '../engine/motune';
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from 'recharts';
import { BarChart3, TrendingUp, ShieldCheck, Scale } from 'lucide-react';

interface DistributionViewerProps {
  prediction: GaussianPrediction;
  activeTarget: TargetType;
}

export const DistributionViewer: React.FC<DistributionViewerProps> = ({
  prediction,
  activeTarget,
}) => {
  const chartData = generateGaussianCurveData(
    prediction.mu,
    prediction.sigma,
    prediction.goldMu,
    prediction.goldSigma
  );

  const getLiteralnessText = (mu: number) => {
    if (mu >= 4.0) return { label: 'High Compositionality (Literal)', color: 'text-emerald-700 bg-emerald-50 border-emerald-200' };
    if (mu >= 2.5) return { label: 'Moderate / Partially Compositional', color: 'text-amber-700 bg-amber-50 border-amber-200' };
    return { label: 'Idiomatic / Figurative Semantic Shift', color: 'text-rose-700 bg-rose-50 border-rose-200' };
  };

  const status = getLiteralnessText(prediction.mu);

  return (
    <div className="bg-white rounded-xl border border-slate-200/80 shadow-xs p-5 space-y-4">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 pb-3 border-b border-slate-100">
        <div className="flex items-center space-x-2">
          <BarChart3 className="w-4 h-4 text-indigo-600" />
          <h3 className="text-xs font-bold uppercase tracking-wider text-slate-800">
            Gaussian Rating & Uncertainty Distribution (<span className="font-mono text-indigo-600">src/heads.py</span>)
          </h3>
        </div>

        <span className={`px-2.5 py-1 rounded-full text-xs font-semibold border ${status.color}`}>
          {status.label}
        </span>
      </div>

      {/* Summary KPI Cards */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
        {/* Predicted Mean mu */}
        <div className="p-3.5 rounded-xl bg-slate-50 border border-slate-200/80">
          <div className="text-[11px] font-semibold text-slate-500 uppercase tracking-wider mb-1 flex items-center justify-between">
            <span>Predicted Mean (μ)</span>
            <TrendingUp className="w-3.5 h-3.5 text-indigo-600" />
          </div>
          <div className="flex items-baseline space-x-2">
            <span className="text-2xl font-bold font-mono text-slate-900">
              {prediction.mu.toFixed(2)}
            </span>
            <span className="text-xs text-slate-500">/ 5.0</span>
          </div>
          {prediction.goldMu !== undefined && (
            <div className="text-[11px] text-slate-500 mt-1">
              Gold benchmark: <strong className="text-slate-800 font-mono">{prediction.goldMu.toFixed(2)}</strong>
            </div>
          )}
        </div>

        {/* Predicted Std sigma */}
        <div className="p-3.5 rounded-xl bg-slate-50 border border-slate-200/80">
          <div className="text-[11px] font-semibold text-slate-500 uppercase tracking-wider mb-1 flex items-center justify-between">
            <span>Uncertainty (σ)</span>
            <ShieldCheck className="w-3.5 h-3.5 text-emerald-600" />
          </div>
          <div className="flex items-baseline space-x-2">
            <span className="text-2xl font-bold font-mono text-slate-900">
              {prediction.sigma.toFixed(2)}
            </span>
            <span className="text-[10px] px-1.5 py-0.2 rounded bg-emerald-50 text-emerald-700 font-mono font-medium">
              &gt; 0.05
            </span>
          </div>
          {prediction.goldSigma !== undefined && (
            <div className="text-[11px] text-slate-500 mt-1">
              Annotator Std: <strong className="text-slate-800 font-mono">{prediction.goldSigma.toFixed(2)}</strong>
            </div>
          )}
        </div>

        {/* 95% Confidence Interval */}
        <div className="p-3.5 rounded-xl bg-slate-50 border border-slate-200/80">
          <div className="text-[11px] font-semibold text-slate-500 uppercase tracking-wider mb-1">
            95% Confidence Interval
          </div>
          <div className="text-lg font-bold font-mono text-slate-800 mt-0.5">
            [{prediction.confidenceInterval[0].toFixed(2)}, {prediction.confidenceInterval[1].toFixed(2)}]
          </div>
          <div className="text-[11px] text-slate-500 mt-1">
            μ ± 1.96σ interval
          </div>
        </div>

        {/* Loss Calibration (KL & CCC) */}
        <div className="p-3.5 rounded-xl bg-slate-50 border border-slate-200/80">
          <div className="text-[11px] font-semibold text-slate-500 uppercase tracking-wider mb-1 flex items-center justify-between">
            <span>Calibration Losses</span>
            <Scale className="w-3.5 h-3.5 text-indigo-500" />
          </div>
          <div className="space-y-1">
            <div className="flex items-center justify-between text-xs font-mono">
              <span className="text-slate-500">KL(Pred || Gold):</span>
              <span className="font-bold text-slate-900">
                {prediction.klDivergence !== undefined ? prediction.klDivergence.toFixed(4) : '—'}
              </span>
            </div>
            <div className="flex items-center justify-between text-xs font-mono">
              <span className="text-slate-500">Lin's CCC:</span>
              <span className="font-bold text-indigo-700">
                {prediction.ccc !== undefined ? prediction.ccc.toFixed(4) : '—'}
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Distribution Curve Chart */}
      <div className="pt-2">
        <div className="text-xs font-semibold text-slate-600 mb-2 flex items-center justify-between">
          <span>Probability Density Function f(x) over Score Range [0.0, 5.0]</span>
          <span className="text-[11px] text-slate-400 font-normal">
            Models human annotator disagreement directly
          </span>
        </div>

        <div className="h-64 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={chartData} margin={{ top: 10, right: 20, left: -20, bottom: 0 }}>
              <defs>
                <linearGradient id="predColor" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#4f46e5" stopOpacity={0.4} />
                  <stop offset="95%" stopColor="#4f46e5" stopOpacity={0.0} />
                </linearGradient>
                <linearGradient id="goldColor" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#10b981" stopOpacity={0.3} />
                  <stop offset="95%" stopColor="#10b981" stopOpacity={0.0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e2e8f0" />
              <XAxis
                dataKey="score"
                tickLine={false}
                stroke="#64748b"
                fontSize={11}
                domain={[0, 5]}
                type="number"
                unit=""
              />
              <YAxis tickLine={false} stroke="#64748b" fontSize={11} />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#ffffff',
                  borderColor: '#cbd5e1',
                  borderRadius: '0.5rem',
                  fontSize: '12px',
                  boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)',
                }}
                formatter={(val: any) => [Number(val).toFixed(4), 'Density']}
                labelFormatter={val => `Score: ${Number(val).toFixed(2)}`}
              />
              <Legend verticalAlign="top" height={36} iconType="circle" wrapperStyle={{ fontSize: '12px' }} />
              <Area
                type="monotone"
                dataKey="predicted"
                name={`Predicted N(μ=${prediction.mu}, σ=${prediction.sigma})`}
                stroke="#4f46e5"
                strokeWidth={2.5}
                fillOpacity={1}
                fill="url(#predColor)"
              />
              {prediction.goldMu !== undefined && (
                <Area
                  type="monotone"
                  dataKey="gold"
                  name={`Gold Benchmark N(y=${prediction.goldMu}, σt=${prediction.goldSigma})`}
                  stroke="#10b981"
                  strokeWidth={2}
                  strokeDasharray="4 4"
                  fillOpacity={1}
                  fill="url(#goldColor)"
                />
              )}
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
};
