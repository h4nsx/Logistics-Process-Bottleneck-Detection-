import { useState, useRef, useEffect } from 'react';
import {
  ChevronRight,
  Play,
  AlertTriangle,
  CheckCircle,
  XCircle,
  Activity,
  BarChart3,
  Clock,
  TrendingUp,
  Info,
  X,
  MapPin,
  Hash,
  Zap,
  FileBarChart,
  Download,
} from 'lucide-react';

/* ──────────────────────────────────────────
   Types
   ────────────────────────────────────────── */

type Severity = 'Normal' | 'Warning' | 'High Risk';

interface DemoRow {
  id: string;
  processId: string;
  stepCode: string;
  location: string;
  expectedDuration: number; // p95, in minutes
  actualDuration: number;   // minutes
  deviation: number;        // actual - mean, minutes
  riskPercent: number;
  severity: Severity;
  // for Explain popover
  baselineMean: number;
  baselineStd: number;
  baselineP95: number;
  zScore: number;
}

/* ──────────────────────────────────────────
   Demo Data — MVP-logic-accurate
   ────────────────────────────────────────── */
// risk_percent = min(100, (actual / p95) * 100)
// anomaly if actual > p95 OR zScore >= 2
// severity: <80 Normal, 80-100 Warning, >100 High Risk

const demoRows: DemoRow[] = [
  {
    id: 'row-1',
    processId: 'ORD-20240301-007',
    stepCode: 'CUSTOMS',
    location: 'HAN-PORT',
    expectedDuration: 180,  // p95 = 180 min
    actualDuration: 284,    // exceeds p95 → High Risk
    deviation: +104,
    riskPercent: 157.8,     // (284/180)*100
    severity: 'High Risk',
    baselineMean: 95,
    baselineStd: 42.5,
    baselineP95: 180,
    zScore: 4.45,           // (284-95)/42.5
  },
  {
    id: 'row-2',
    processId: 'ORD-20240301-012',
    stepCode: 'LAST_MILE',
    location: 'HCM-DIST-3',
    expectedDuration: 95,   // p95
    actualDuration: 81,     // 85.3% → Warning
    deviation: +18,
    riskPercent: 85.3,      // (81/95)*100
    severity: 'Warning',
    baselineMean: 55,
    baselineStd: 18.2,
    baselineP95: 95,
    zScore: 1.43,           // (81-55)/18.2 — below 2 but near p95 → Warning via risk%
  },
  {
    id: 'row-3',
    processId: 'ORD-20240301-003',
    stepCode: 'PICK',
    location: 'HAN-WH-01',
    expectedDuration: 55,   // p95
    actualDuration: 38,     // 69.1% → Normal
    deviation: -5,
    riskPercent: 69.1,      // (38/55)*100
    severity: 'Normal',
    baselineMean: 28,
    baselineStd: 9.1,
    baselineP95: 55,
    zScore: 1.1,
  },
  {
    id: 'row-4',
    processId: 'ORD-20240301-019',
    stepCode: 'PACK',
    location: 'HAN-WH-02',
    expectedDuration: 42,   // p95
    actualDuration: 40,     // 95.2% → Warning
    deviation: +12,
    riskPercent: 95.2,      // (40/42)*100
    severity: 'Warning',
    baselineMean: 22,
    baselineStd: 8.8,
    baselineP95: 42,
    zScore: 2.05,           // (40-22)/8.8 → ≥2, also anomaly by z-score
  },
];

/* ──────────────────────────────────────────
   Severity Badge
   ────────────────────────────────────────── */

const SeverityBadge = ({ severity }: { severity: Severity }) => {
  const config: Record<Severity, { icon: React.ReactNode; cls: string }> = {
    Normal: {
      icon: <CheckCircle className="w-3.5 h-3.5" />,
      cls: 'bg-success/10 text-success border-success/20',
    },
    Warning: {
      icon: <AlertTriangle className="w-3.5 h-3.5" />,
      cls: 'bg-warning/10 text-warning border-warning/20',
    },
    'High Risk': {
      icon: <XCircle className="w-3.5 h-3.5" />,
      cls: 'bg-danger/10 text-danger border-danger/20',
    },
  };
  const { icon, cls } = config[severity];
  return (
    <span
      className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-semibold border ${cls}`}
    >
      {icon}
      {severity}
    </span>
  );
};

/* ──────────────────────────────────────────
   Risk % Bar
   ────────────────────────────────────────── */

const RiskBar = ({
  value,
  severity,
}: {
  value: number;
  severity: Severity;
}) => {
  const barColor =
    severity === 'High Risk'
      ? 'bg-danger'
      : severity === 'Warning'
      ? 'bg-warning'
      : 'bg-success';

  // clamp width at 100% visually
  const widthPct = Math.min(100, value);

  return (
    <div className="flex items-center gap-2 min-w-[100px]">
      <div className="flex-1 h-1.5 rounded-full bg-surface-dark overflow-hidden">
        <div
          className={`h-full rounded-full ${barColor} transition-all duration-700`}
          style={{ width: `${widthPct}%` }}
        />
      </div>
      <span
        className={`text-xs font-semibold tabular-nums ${
          severity === 'High Risk'
            ? 'text-danger'
            : severity === 'Warning'
            ? 'text-warning'
            : 'text-success'
        }`}
      >
        {value.toFixed(1)}%
      </span>
    </div>
  );
};

/* ──────────────────────────────────────────
   Explain Popover
   ────────────────────────────────────────── */

const ExplainPopover = ({
  row,
  onClose,
}: {
  row: DemoRow;
  onClose: () => void;
}) => {
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) onClose();
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, [onClose]);

  const stats = [
    {
      label: 'Baseline Mean (μ)',
      value: `${row.baselineMean} min`,
      sub: 'Average across all historical runs',
      color: 'text-cyan',
      bg: 'bg-cyan/10',
    },
    {
      label: 'Std Deviation (σ)',
      value: `±${row.baselineStd} min`,
      sub: 'Typical spread around the mean',
      color: 'text-orange',
      bg: 'bg-orange/10',
    },
    {
      label: 'Baseline p95',
      value: `${row.baselineP95} min`,
      sub: '95th percentile — upper "normal" ceiling',
      color: 'text-navy',
      bg: 'bg-navy/10',
    },
    {
      label: 'Actual Duration',
      value: `${row.actualDuration} min`,
      sub: `+${(row.actualDuration - row.baselineP95)} min over p95`,
      color: 'text-danger',
      bg: 'bg-danger/10',
    },
    {
      label: 'Z-Score',
      value: row.zScore.toFixed(2),
      sub: row.zScore >= 2 ? '≥ 2 — statistically significant anomaly' : '< 2',
      color: row.zScore >= 2 ? 'text-danger' : 'text-content-secondary',
      bg: row.zScore >= 2 ? 'bg-danger/10' : 'bg-surface',
    },
  ];

  return (
    <div
      ref={ref}
      className="absolute right-0 top-full mt-2 z-50 w-72 bg-white rounded-2xl border border-border shadow-dropdown animate-slide-down"
    >
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-border bg-danger/5 rounded-t-2xl">
        <div className="flex items-center gap-2">
          <Zap className="w-4 h-4 text-danger" />
          <span className="text-sm font-bold text-danger">Bottleneck Explained</span>
        </div>
        <button
          onClick={onClose}
          className="text-content-muted hover:text-content-secondary transition-colors"
        >
          <X className="w-4 h-4" />
        </button>
      </div>

      {/* Step info */}
      <div className="px-4 pt-3 pb-1">
        <p className="text-xs text-content-muted mb-0.5">Step</p>
        <div className="flex items-center gap-2">
          <code className="text-sm font-mono font-bold text-navy">
            {row.stepCode}
          </code>
          <span className="text-content-muted text-xs">@</span>
          <span className="text-xs text-content-secondary">{row.location}</span>
        </div>
      </div>

      {/* Stats grid */}
      <div className="px-4 py-3 space-y-2">
        {stats.map(({ label, value, sub, color, bg }) => (
          <div
            key={label}
            className={`flex items-center justify-between rounded-lg ${bg} px-3 py-2.5`}
          >
            <div>
              <p className="text-[11px] font-semibold text-content-secondary">
                {label}
              </p>
              <p className="text-[10px] text-content-muted leading-tight mt-0.5">
                {sub}
              </p>
            </div>
            <span className={`text-sm font-bold font-mono ${color}`}>
              {value}
            </span>
          </div>
        ))}
      </div>

      {/* Why flagged */}
      <div className="px-4 pb-4">
        <div className="bg-navy rounded-xl px-3 py-2.5">
          <p className="text-[10px] font-semibold text-orange uppercase tracking-wider mb-1">
            Why flagged?
          </p>
          <p className="text-[11px] text-white/80 leading-relaxed">
            Actual duration ({row.actualDuration} min) exceeds p95 (
            {row.baselineP95} min) and z-score ({row.zScore.toFixed(2)}) is{' '}
            {row.zScore >= 2 ? '≥ 2' : '< 2'}. Either condition alone triggers
            a High Risk alert.
          </p>
        </div>
      </div>
    </div>
  );
};

/* ──────────────────────────────────────────
   KPI Summary Card
   ────────────────────────────────────────── */

const KpiCard = ({
  label,
  value,
  sub,
  icon,
  accent,
}: {
  label: string;
  value: string | number;
  sub: string;
  icon: React.ReactNode;
  accent: string;
}) => (
  <div className="bg-white rounded-2xl border border-border shadow-card p-5 flex items-start gap-4">
    <div
      className={`w-11 h-11 rounded-xl flex items-center justify-center shrink-0 ${accent}`}
    >
      {icon}
    </div>
    <div className="min-w-0">
      <p className="text-xs font-medium text-content-muted uppercase tracking-wider mb-1">
        {label}
      </p>
      <p className="text-2xl font-black text-navy leading-none">{value}</p>
      <p className="text-xs text-content-secondary mt-1">{sub}</p>
    </div>
  </div>
);

/* ──────────────────────────────────────────
   Main Interactive Demo Page
   ────────────────────────────────────────── */

const InteractiveDemoPage = () => {
  const [explainRowId, setExplainRowId] = useState<string | null>(null);
  const [filterSeverity, setFilterSeverity] = useState<Severity | 'All'>('All');
  const [highlightNew, setHighlightNew] = useState(false);

  const filtered =
    filterSeverity === 'All'
      ? demoRows
      : demoRows.filter((r) => r.severity === filterSeverity);

  const highRiskCount = demoRows.filter((r) => r.severity === 'High Risk').length;
  const warningCount = demoRows.filter((r) => r.severity === 'Warning').length;
  const avgRisk =
    demoRows.reduce((s, r) => s + r.riskPercent, 0) / demoRows.length;

  // Simulate "running analysis"
  const handleRunDemo = () => {
    setHighlightNew(true);
    setTimeout(() => setHighlightNew(false), 2000);
  };

  return (
    <div className="min-h-screen bg-surface pt-20">

      {/* ── Page Header ── */}
      <div className="bg-white border-b border-border">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-10">
          {/* Breadcrumb */}
          <div className="flex items-center gap-2 text-sm text-content-muted mb-3">
            <FileBarChart className="w-4 h-4" />
            <span>Demo</span>
            <ChevronRight className="w-3 h-3" />
            <span className="text-navy font-medium">Example Analysis</span>
          </div>

          <div className="flex flex-col sm:flex-row sm:items-start sm:justify-between gap-4">
            <div>
              <h1 className="text-3xl sm:text-4xl font-bold text-navy mb-2">
                Bottleneck Dashboard
              </h1>
              <p className="text-content-secondary max-w-xl">
                A live mock of what Vyn's analysis looks like after processing
                your logistics data. All values reflect real MVP detection logic
                — hover the{' '}
                <strong className="text-danger">High Risk</strong> rows and
                click <strong>Explain</strong> to see the statistical breakdown.
              </p>
            </div>

            {/* Action buttons */}
            <div className="flex items-center gap-2 shrink-0">
              <button
                onClick={handleRunDemo}
                className="inline-flex items-center gap-2 px-4 py-2.5 rounded-xl bg-orange text-white text-sm font-semibold hover:bg-orange-dark active:scale-[0.97] transition-all shadow-sm hover:shadow-md"
              >
                <Play className="w-4 h-4" />
                Run Analysis
              </button>
              <button className="inline-flex items-center gap-2 px-4 py-2.5 rounded-xl bg-white border border-border text-content-secondary text-sm font-medium hover:border-border-dark hover:text-navy transition-all">
                <Download className="w-4 h-4" />
                Export CSV
              </button>
            </div>
          </div>

          {/* Demo tag */}
          <div className="flex items-center gap-2 mt-4">
            <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold bg-orange/10 text-orange border border-orange/20">
              <Activity className="w-3 h-3" />
              Live Demo — Sample Dataset
            </span>
            <span className="text-xs text-content-muted">
              {demoRows.length} step executions analysed
            </span>
          </div>
        </div>
      </div>

      {/* ── Body ── */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-8">

        {/* ══ KPI Summary Cards ══ */}
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
          <KpiCard
            label="Total Steps"
            value={demoRows.length}
            sub="step executions in this dataset"
            icon={<BarChart3 className="w-5 h-5 text-cyan" />}
            accent="bg-cyan/10"
          />
          <KpiCard
            label="High Risk"
            value={highRiskCount}
            sub="active bottlenecks detected"
            icon={<XCircle className="w-5 h-5 text-danger" />}
            accent="bg-danger/10"
          />
          <KpiCard
            label="Warnings"
            value={warningCount}
            sub="steps approaching baseline ceiling"
            icon={<AlertTriangle className="w-5 h-5 text-warning" />}
            accent="bg-warning/10"
          />
          <KpiCard
            label="Avg. Risk Score"
            value={`${avgRisk.toFixed(1)}%`}
            sub="across all steps in dataset"
            icon={<TrendingUp className="w-5 h-5 text-orange" />}
            accent="bg-orange/10"
          />
        </div>

        {/* ══ Data Table Section ══ */}
        <div className="bg-white rounded-2xl border border-border shadow-card overflow-hidden">

          {/* Table toolbar */}
          <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3 px-5 py-4 border-b border-border">
            <div>
              <h2 className="text-base font-bold text-navy">
                Step Execution Analysis
              </h2>
              <p className="text-xs text-content-muted mt-0.5">
                Showing {filtered.length} of {demoRows.length} records
              </p>
            </div>

            {/* Severity filter */}
            <div className="flex flex-wrap gap-2">
              {(['All', 'Normal', 'Warning', 'High Risk'] as const).map((s) => (
                <button
                  key={s}
                  onClick={() => setFilterSeverity(s)}
                  className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${
                    filterSeverity === s
                      ? s === 'High Risk'
                        ? 'bg-danger text-white'
                        : s === 'Warning'
                        ? 'bg-warning text-white'
                        : s === 'Normal'
                        ? 'bg-success text-white'
                        : 'bg-navy text-white'
                      : 'bg-surface text-content-secondary hover:bg-surface-dark border border-border'
                  }`}
                >
                  {s}
                </button>
              ))}
            </div>
          </div>

          {/* Scrollable table */}
          <div className="overflow-x-auto">
            <table className="w-full text-sm min-w-[800px]">
              <thead>
                <tr className="bg-surface border-b border-border">
                  {[
                    { label: 'Process ID', icon: <Hash className="w-3.5 h-3.5" /> },
                    { label: 'Step Code', icon: null },
                    { label: 'Location', icon: <MapPin className="w-3.5 h-3.5" /> },
                    { label: 'Expected (p95)', icon: <Clock className="w-3.5 h-3.5" /> },
                    { label: 'Actual Duration', icon: <Activity className="w-3.5 h-3.5" /> },
                    { label: 'Deviation', icon: <TrendingUp className="w-3.5 h-3.5" /> },
                    { label: 'Risk %', icon: <BarChart3 className="w-3.5 h-3.5" /> },
                    { label: 'Severity', icon: null },
                    { label: '', icon: null }, // Explain button column
                  ].map(({ label, icon }) => (
                    <th
                      key={label}
                      className="text-left px-4 py-3 text-[11px] font-semibold uppercase tracking-wider text-content-muted whitespace-nowrap"
                    >
                      {label && (
                        <span className="flex items-center gap-1.5">
                          {icon}
                          {label}
                        </span>
                      )}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody className="divide-y divide-border">
                {filtered.map((row, i) => {
                  const isHighRisk = row.severity === 'High Risk';
                  const isExplainOpen = explainRowId === row.id;

                  return (
                    <tr
                      key={row.id}
                      className={`transition-colors group ${
                        highlightNew && i === 0
                          ? 'animate-pulse bg-orange/5'
                          : isHighRisk
                          ? 'bg-danger/[0.025] hover:bg-danger/[0.05]'
                          : 'hover:bg-surface/60'
                      }`}
                    >
                      {/* Process ID */}
                      <td className="px-4 py-3.5">
                        <code className="font-mono text-xs font-semibold text-navy bg-navy/5 px-2 py-1 rounded">
                          {row.processId}
                        </code>
                      </td>

                      {/* Step Code */}
                      <td className="px-4 py-3.5">
                        <span className="inline-block px-2 py-0.5 rounded text-xs font-bold bg-cyan/10 text-cyan-dark font-mono">
                          {row.stepCode}
                        </span>
                      </td>

                      {/* Location */}
                      <td className="px-4 py-3.5">
                        <span className="flex items-center gap-1.5 text-xs text-content-secondary">
                          <MapPin className="w-3.5 h-3.5 text-content-muted shrink-0" />
                          {row.location}
                        </span>
                      </td>

                      {/* Expected Duration (p95) */}
                      <td className="px-4 py-3.5 tabular-nums text-content-secondary text-xs">
                        {row.expectedDuration} min
                      </td>

                      {/* Actual Duration */}
                      <td className="px-4 py-3.5 tabular-nums">
                        <span
                          className={`text-sm font-bold ${
                            isHighRisk ? 'text-danger' : 'text-content-primary'
                          }`}
                        >
                          {row.actualDuration} min
                        </span>
                      </td>

                      {/* Deviation */}
                      <td className="px-4 py-3.5 tabular-nums">
                        <span
                          className={`text-xs font-semibold ${
                            row.deviation > 0
                              ? 'text-danger'
                              : 'text-success'
                          }`}
                        >
                          {row.deviation > 0 ? '+' : ''}
                          {row.deviation} min
                        </span>
                      </td>

                      {/* Risk % with bar */}
                      <td className="px-4 py-3.5 min-w-[130px]">
                        <RiskBar
                          value={row.riskPercent}
                          severity={row.severity}
                        />
                      </td>

                      {/* Severity badge */}
                      <td className="px-4 py-3.5">
                        <SeverityBadge severity={row.severity} />
                      </td>

                      {/* Explain button — only for High Risk */}
                      <td className="px-4 py-3.5">
                        <div className="relative">
                          {isHighRisk ? (
                            <button
                              onClick={() =>
                                setExplainRowId(isExplainOpen ? null : row.id)
                              }
                              className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold transition-all border ${
                                isExplainOpen
                                  ? 'bg-navy text-white border-navy'
                                  : 'bg-danger/10 text-danger border-danger/20 hover:bg-danger/20'
                              }`}
                            >
                              <Info className="w-3.5 h-3.5" />
                              Explain
                            </button>
                          ) : (
                            <span className="text-xs text-content-muted px-2">
                              —
                            </span>
                          )}

                          {/* Popover */}
                          {isExplainOpen && (
                            <ExplainPopover
                              row={row}
                              onClose={() => setExplainRowId(null)}
                            />
                          )}
                        </div>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>

          {/* Table footer */}
          <div className="flex items-center justify-between px-5 py-3 border-t border-border bg-surface/50">
            <p className="text-xs text-content-muted">
              Sample dataset · 4 step executions across 2 warehouses, 1 port
            </p>
            <div className="flex items-center gap-4 text-xs text-content-muted">
              <span className="flex items-center gap-1">
                <span className="w-2 h-2 rounded-full bg-success" />
                Normal
              </span>
              <span className="flex items-center gap-1">
                <span className="w-2 h-2 rounded-full bg-warning" />
                Warning
              </span>
              <span className="flex items-center gap-1">
                <span className="w-2 h-2 rounded-full bg-danger" />
                High Risk
              </span>
            </div>
          </div>
        </div>

        {/* ══ Detection Logic Reference ══ */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* How risk% is calculated */}
          <div className="bg-white rounded-2xl border border-border shadow-card p-6">
            <h3 className="text-sm font-bold text-navy mb-1 flex items-center gap-2">
              <BarChart3 className="w-4 h-4 text-orange" />
              Risk % Formula
            </h3>
            <p className="text-xs text-content-muted mb-4">
              How each row's risk percentage is calculated from the data above.
            </p>
            <div className="bg-navy rounded-xl px-4 py-3 font-mono text-sm text-white text-center mb-4">
              risk_percent = min(100, (actual / p95) × 100)
            </div>
            <div className="space-y-2">
              {[
                { label: 'CUSTOMS', actual: 284, p95: 180, result: '157.8%', color: 'text-danger' },
                { label: 'LAST_MILE', actual: 81, p95: 95, result: '85.3%', color: 'text-warning' },
                { label: 'PICK', actual: 38, p95: 55, result: '69.1%', color: 'text-success' },
                { label: 'PACK', actual: 40, p95: 42, result: '95.2%', color: 'text-warning' },
              ].map(({ label, actual, p95, result, color }) => (
                <div
                  key={label}
                  className="flex items-center justify-between text-xs bg-surface rounded-lg px-3 py-2"
                >
                  <code className="font-mono font-semibold text-navy">{label}</code>
                  <span className="text-content-muted font-mono">
                    ({actual}/{p95}) × 100
                  </span>
                  <span className={`font-bold font-mono ${color}`}>{result}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Anomaly trigger explanation */}
          <div className="bg-white rounded-2xl border border-border shadow-card p-6">
            <h3 className="text-sm font-bold text-navy mb-1 flex items-center gap-2">
              <Zap className="w-4 h-4 text-danger" />
              Anomaly Detection Rules
            </h3>
            <p className="text-xs text-content-muted mb-4">
              A step is flagged if <strong>either</strong> condition is true.
            </p>
            <div className="space-y-3">
              <div className="border border-danger/20 bg-danger/5 rounded-xl p-4">
                <p className="text-xs font-bold text-danger mb-1">
                  Rule 1 — Exceeds p95
                </p>
                <code className="text-xs font-mono text-navy block bg-white rounded px-3 py-2 mb-2">
                  actual_duration {'>'} p95
                </code>
                <p className="text-[11px] text-content-muted">
                  CUSTOMS: 284 {'>'} 180 ✓ — immediately flagged
                </p>
              </div>
              <div className="border border-orange/20 bg-orange/5 rounded-xl p-4">
                <p className="text-xs font-bold text-orange mb-1">
                  Rule 2 — Z-score ≥ 2
                </p>
                <code className="text-xs font-mono text-navy block bg-white rounded px-3 py-2 mb-2">
                  (actual − mean) / std ≥ 2
                </code>
                <p className="text-[11px] text-content-muted">
                  PACK: (40−22)/8.8 = 2.05 ≥ 2 ✓ — also flagged
                </p>
              </div>
            </div>
          </div>
        </div>

        {/* ── CTA Banner ── */}
        <div className="bg-navy rounded-2xl p-8 flex flex-col sm:flex-row items-center justify-between gap-6">
          <div>
            <h3 className="text-xl font-bold text-white mb-1">
              Ready to analyse your own data?
            </h3>
            <p className="text-navy-100 text-sm">
              Upload your logistics CSV and get a real bottleneck report in
              seconds.
            </p>
          </div>
          <div className="flex gap-3 shrink-0">
            <a
              href="/resources/docs"
              className="inline-flex items-center gap-2 px-4 py-2.5 rounded-xl bg-white/10 hover:bg-white/20 text-white text-sm font-semibold transition-colors border border-white/20"
            >
              Read Docs
            </a>
            <a
              href="/register"
              className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl bg-orange hover:bg-orange-dark text-white text-sm font-bold transition-colors shadow-sm"
            >
              Get Started Free
              <ChevronRight className="w-4 h-4" />
            </a>
          </div>
        </div>

      </div>
    </div>
  );
};

export default InteractiveDemoPage;
