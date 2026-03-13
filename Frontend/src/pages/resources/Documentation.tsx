import { useState, useEffect, useRef } from 'react';
import {
  BookOpen,
  Database,
  Cpu,
  ShieldCheck,
  ChevronRight,
  AlertTriangle,
  CheckCircle,
  XCircle,
  Clock,
  Hash,
  MapPin,
  Calendar,
  ArrowUp,
  Info,
  FileText,
  BarChart3,
  Zap,
} from 'lucide-react';

/* ──────────────────────────────────────────
   Types
   ────────────────────────────────────────── */

interface SidebarSection {
  id: string;
  label: string;
  icon: React.ReactNode;
  subsections?: { id: string; label: string }[];
}

/* ──────────────────────────────────────────
   Sidebar Navigation Data
   ────────────────────────────────────────── */

const sections: SidebarSection[] = [
  {
    id: 'introduction',
    label: 'Introduction',
    icon: <BookOpen className="w-4 h-4" />,
  },
  {
    id: 'data-preparation',
    label: 'Data Preparation',
    icon: <Database className="w-4 h-4" />,
    subsections: [
      { id: 'required-schema', label: 'Required Schema' },
      { id: 'data-validation', label: 'Data Validation' },
    ],
  },
  {
    id: 'core-logic',
    label: 'Core Logic',
    icon: <Cpu className="w-4 h-4" />,
    subsections: [
      { id: 'baseline-statistics', label: 'Baseline Statistics' },
      { id: 'detection-rules', label: 'Detection Rules' },
    ],
  },
  {
    id: 'risk-scoring',
    label: 'Risk Scoring',
    icon: <ShieldCheck className="w-4 h-4" />,
    subsections: [
      { id: 'risk-categories', label: 'Risk Categories' },
      { id: 'risk-formula', label: 'Scoring Formula' },
    ],
  },
];

/* ──────────────────────────────────────────
   Schema Field Data
   ────────────────────────────────────────── */

const schemaFields = [
  {
    field: 'process_id',
    type: 'string',
    icon: <Hash className="w-4 h-4 text-cyan" />,
    required: true,
    description: 'Unique identifier for the logistics process instance (e.g., shipment ID, order number).',
    example: '"ORD-20240301-001"',
  },
  {
    field: 'step_code',
    type: 'string',
    icon: <FileText className="w-4 h-4 text-cyan" />,
    required: true,
    description: 'Short code identifying the specific process step (e.g., PICK, PACK, SHIP, CUSTOMS).',
    example: '"PICK"',
  },
  {
    field: 'start_time',
    type: 'datetime',
    icon: <Calendar className="w-4 h-4 text-orange" />,
    required: true,
    description: 'ISO 8601 timestamp when this step began. Must be earlier than end_time.',
    example: '"2024-03-01T08:00:00Z"',
  },
  {
    field: 'end_time',
    type: 'datetime',
    icon: <Calendar className="w-4 h-4 text-orange" />,
    required: true,
    description: 'ISO 8601 timestamp when this step completed. Must be later than start_time.',
    example: '"2024-03-01T08:45:00Z"',
  },
  {
    field: 'location',
    type: 'string',
    icon: <MapPin className="w-4 h-4 text-navy" />,
    required: true,
    description: 'Facility or geographic location where this step occurred (e.g., warehouse code, port name).',
    example: '"HAN-WH-01"',
  },
];

/* ──────────────────────────────────────────
   Risk Category Data
   ────────────────────────────────────────── */

const riskCategories = [
  {
    level: 'Normal',
    threshold: '< 80%',
    icon: <CheckCircle className="w-5 h-5" />,
    colorClass: 'text-success',
    bgClass: 'bg-success-50',
    borderClass: 'border-success/20',
    badgeClass: 'bg-success/10 text-success',
    description: 'Step duration is well within the historical normal range. No action required.',
    condition: 'risk_percent < 80',
  },
  {
    level: 'Warning',
    threshold: '80% – 100%',
    icon: <AlertTriangle className="w-5 h-5" />,
    colorClass: 'text-warning',
    bgClass: 'bg-warning-50',
    borderClass: 'border-warning/20',
    badgeClass: 'bg-warning/10 text-warning',
    description: 'Step duration is approaching the 95th percentile limit. Monitor closely.',
    condition: '80 ≤ risk_percent ≤ 100',
  },
  {
    level: 'High Risk',
    threshold: '> 100%',
    icon: <XCircle className="w-5 h-5" />,
    colorClass: 'text-danger',
    bgClass: 'bg-danger-50',
    borderClass: 'border-danger/20',
    badgeClass: 'bg-danger/10 text-danger',
    description: 'Step duration has exceeded the 95th percentile baseline. This is an active bottleneck.',
    condition: 'risk_percent > 100',
  },
];

/* ──────────────────────────────────────────
   Section wrapper with scroll-margin
   ────────────────────────────────────────── */

const Section = ({
  id,
  children,
}: {
  id: string;
  children: React.ReactNode;
}) => (
  <section id={id} className="scroll-mt-24">
    {children}
  </section>
);

/* ──────────────────────────────────────────
   Main Documentation Page
   ────────────────────────────────────────── */

const DocumentationPage = () => {
  const [activeSection, setActiveSection] = useState('introduction');
  const [showScrollTop, setShowScrollTop] = useState(false);
  const contentRef = useRef<HTMLDivElement>(null);

  /* Active section tracking via IntersectionObserver */
  useEffect(() => {
    const allIds = sections.flatMap((s) =>
      s.subsections ? [s.id, ...s.subsections.map((sub) => sub.id)] : [s.id]
    );

    const observers: IntersectionObserver[] = [];

    allIds.forEach((id) => {
      const el = document.getElementById(id);
      if (!el) return;
      const obs = new IntersectionObserver(
        ([entry]) => {
          if (entry.isIntersecting) setActiveSection(id);
        },
        { rootMargin: '-20% 0px -70% 0px', threshold: 0 }
      );
      obs.observe(el);
      observers.push(obs);
    });

    return () => observers.forEach((o) => o.disconnect());
  }, []);

  /* Scroll-to-top button */
  useEffect(() => {
    const handleScroll = () => setShowScrollTop(window.scrollY > 400);
    window.addEventListener('scroll', handleScroll, { passive: true });
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  const scrollTo = (id: string) => {
    document.getElementById(id)?.scrollIntoView({ behavior: 'smooth' });
  };

  return (
    <div className="min-h-screen bg-surface pt-20">
      {/* ── Page Header ── */}
      <div className="bg-white border-b border-border">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-10">
          <div className="flex items-center gap-2 text-sm text-content-muted mb-3">
            <BookOpen className="w-4 h-4" />
            <span>Resources</span>
            <ChevronRight className="w-3 h-3" />
            <span className="text-navy font-medium">Documentation</span>
          </div>
          <h1 className="text-3xl sm:text-4xl font-bold text-navy mb-3">
            Documentation
          </h1>
          <p className="text-content-secondary text-lg max-w-2xl">
            Everything you need to understand, prepare, and analyse your
            logistics data using the Vyn bottleneck detection platform.
          </p>
          {/* Quick-stat pills */}
          <div className="flex flex-wrap gap-3 mt-6">
            {[
              { icon: <Database className="w-3.5 h-3.5" />, label: '5 Required Fields' },
              { icon: <BarChart3 className="w-3.5 h-3.5" />, label: 'CSV Upload Only' },
              { icon: <Zap className="w-3.5 h-3.5" />, label: 'Statistical Detection' },
            ].map(({ icon, label }) => (
              <span
                key={label}
                className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-medium bg-navy/5 text-navy border border-navy/10"
              >
                {icon}
                {label}
              </span>
            ))}
          </div>
        </div>
      </div>

      {/* ── Body: Sidebar + Content ── */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-10">
        <div className="flex gap-8 lg:gap-12 items-start">

          {/* ─── Sidebar ─── */}
          <aside className="hidden lg:block w-60 xl:w-64 shrink-0 sticky top-24 self-start">
            <nav className="bg-white rounded-xl border border-border shadow-card p-4">
              <p className="text-[10px] font-semibold uppercase tracking-widest text-content-muted mb-3 px-2">
                On this page
              </p>
              <ul className="space-y-0.5">
                {sections.map((section) => (
                  <li key={section.id}>
                    <button
                      onClick={() => scrollTo(section.id)}
                      className={`flex items-center gap-2.5 w-full text-left px-3 py-2 rounded-lg text-sm font-medium transition-colors duration-150 ${
                        activeSection === section.id
                          ? 'bg-orange/10 text-orange'
                          : 'text-content-secondary hover:bg-surface hover:text-navy'
                      }`}
                    >
                      <span
                        className={
                          activeSection === section.id
                            ? 'text-orange'
                            : 'text-content-muted'
                        }
                      >
                        {section.icon}
                      </span>
                      {section.label}
                    </button>

                    {section.subsections && (
                      <ul className="ml-4 mt-0.5 space-y-0.5 pl-3 border-l border-border">
                        {section.subsections.map((sub) => (
                          <li key={sub.id}>
                            <button
                              onClick={() => scrollTo(sub.id)}
                              className={`block w-full text-left px-2 py-1.5 rounded-md text-xs font-medium transition-colors duration-150 ${
                                activeSection === sub.id
                                  ? 'text-orange'
                                  : 'text-content-muted hover:text-content-secondary'
                              }`}
                            >
                              {sub.label}
                            </button>
                          </li>
                        ))}
                      </ul>
                    )}
                  </li>
                ))}
              </ul>
            </nav>
          </aside>

          {/* ─── Main Content ─── */}
          <div ref={contentRef} className="flex-1 min-w-0 space-y-14">

            {/* ═══════════════════════════════
                1. INTRODUCTION
                ═══════════════════════════════ */}
            <Section id="introduction">
              <div className="flex items-center gap-3 mb-6">
                <div className="w-10 h-10 rounded-xl bg-navy/10 flex items-center justify-center shrink-0">
                  <BookOpen className="w-5 h-5 text-navy" />
                </div>
                <div>
                  <h2 className="text-2xl font-bold text-navy">Introduction</h2>
                  <p className="text-sm text-content-muted">What Vyn does for your supply chain</p>
                </div>
              </div>

              <div className="prose-like space-y-4 text-content-secondary leading-relaxed">
                <p>
                  Vyn is a{' '}
                  <strong className="text-navy">
                    logistics process bottleneck detection platform
                  </strong>{' '}
                  designed to help operations teams identify exactly where and
                  when their supply chain slows down — before delays cascade into
                  costly disruptions.
                </p>
                <p>
                  Every logistics process is broken into discrete{' '}
                  <em>steps</em> (e.g., pick, pack, customs clearance, last-mile
                  delivery). Vyn analyses the duration of each step across
                  thousands of historical executions and automatically flags the
                  ones that deviate from normal behaviour.
                </p>

                {/* Highlight card */}
                <div className="bg-navy rounded-xl p-5 text-white mt-6">
                  <div className="flex items-start gap-3">
                    <Zap className="w-5 h-5 text-orange shrink-0 mt-0.5" />
                    <div>
                      <p className="font-semibold text-white mb-1">
                        No guesswork. Pure statistics.
                      </p>
                      <p className="text-navy-100 text-sm leading-relaxed">
                        The system builds a statistical baseline from your own
                        historical data — calculating the mean, standard
                        deviation, and 95th percentile (p95) for every step. A
                        step is flagged as an anomaly only when its duration
                        significantly exceeds these learned boundaries.
                      </p>
                    </div>
                  </div>
                </div>

                {/* Scope callout */}
                <div className="bg-orange-50 border border-orange/20 rounded-xl p-4 mt-4">
                  <div className="flex items-start gap-2.5">
                    <Info className="w-4 h-4 text-orange shrink-0 mt-0.5" />
                    <div>
                      <p className="text-sm font-semibold text-orange mb-1">
                        Current Scope
                      </p>
                      <p className="text-sm text-content-secondary">
                        Vyn uses <strong>statistical intelligence</strong> based
                        on historical execution data. It does not make
                        predictions about future events, does not perform causal
                        analysis, and does not connect to external data sources.
                        Data is ingested via{' '}
                        <strong>CSV file upload only</strong>.
                      </p>
                    </div>
                  </div>
                </div>

                {/* How it works - 3-step */}
                <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 mt-6">
                  {[
                    {
                      step: '01',
                      title: 'Upload your CSV',
                      desc: 'Provide historical execution records in the required schema.',
                      color: 'text-cyan',
                      bg: 'bg-cyan/10',
                    },
                    {
                      step: '02',
                      title: 'Baseline is built',
                      desc: 'The engine computes mean, std, and p95 per step automatically.',
                      color: 'text-orange',
                      bg: 'bg-orange/10',
                    },
                    {
                      step: '03',
                      title: 'Bottlenecks flagged',
                      desc: 'Anomalous steps are highlighted with a risk score.',
                      color: 'text-navy',
                      bg: 'bg-navy/10',
                    },
                  ].map(({ step, title, desc, color, bg }) => (
                    <div
                      key={step}
                      className="bg-white rounded-xl border border-border p-4 shadow-card"
                    >
                      <span
                        className={`inline-block text-xs font-bold px-2 py-0.5 rounded-md ${bg} ${color} mb-3`}
                      >
                        STEP {step}
                      </span>
                      <p className="font-semibold text-navy text-sm mb-1">
                        {title}
                      </p>
                      <p className="text-xs text-content-muted leading-relaxed">
                        {desc}
                      </p>
                    </div>
                  ))}
                </div>
              </div>
            </Section>

            {/* ═══════════════════════════════
                2. DATA PREPARATION
                ═══════════════════════════════ */}
            <Section id="data-preparation">
              <div className="flex items-center gap-3 mb-6">
                <div className="w-10 h-10 rounded-xl bg-cyan/10 flex items-center justify-center shrink-0">
                  <Database className="w-5 h-5 text-cyan" />
                </div>
                <div>
                  <h2 className="text-2xl font-bold text-navy">
                    Data Preparation
                  </h2>
                  <p className="text-sm text-content-muted">
                    How to format your CSV before uploading
                  </p>
                </div>
              </div>

              <p className="text-content-secondary mb-6 leading-relaxed">
                Vyn processes data exclusively through CSV file upload. Before
                uploading, ensure your file contains the five required columns
                described below. Column names are{' '}
                <strong className="text-navy">case-sensitive</strong> and must
                match exactly.
              </p>

              {/* ─ Required Schema ─ */}
              <Section id="required-schema">
                <h3 className="text-lg font-semibold text-navy mb-4 flex items-center gap-2">
                  <Hash className="w-4 h-4 text-cyan" />
                  Required Schema
                </h3>

                {/* Schema Table */}
                <div className="bg-white rounded-xl border border-border shadow-card overflow-hidden mb-6">
                  <div className="overflow-x-auto">
                    <table className="w-full text-sm">
                      <thead>
                        <tr className="bg-navy text-white">
                          <th className="text-left px-4 py-3 font-semibold text-xs uppercase tracking-wider">
                            Field Name
                          </th>
                          <th className="text-left px-4 py-3 font-semibold text-xs uppercase tracking-wider">
                            Type
                          </th>
                          <th className="text-left px-4 py-3 font-semibold text-xs uppercase tracking-wider hidden md:table-cell">
                            Description
                          </th>
                          <th className="text-left px-4 py-3 font-semibold text-xs uppercase tracking-wider hidden lg:table-cell">
                            Example Value
                          </th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-border">
                        {schemaFields.map((field, i) => (
                          <tr
                            key={field.field}
                            className={`transition-colors hover:bg-surface/60 ${
                              i % 2 === 0 ? 'bg-white' : 'bg-surface/30'
                            }`}
                          >
                            <td className="px-4 py-3.5">
                              <div className="flex items-center gap-2">
                                {field.icon}
                                <code className="font-mono font-semibold text-navy text-xs bg-navy/5 px-2 py-0.5 rounded">
                                  {field.field}
                                </code>
                              </div>
                            </td>
                            <td className="px-4 py-3.5">
                              <span
                                className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[11px] font-medium ${
                                  field.type === 'datetime'
                                    ? 'bg-orange-50 text-orange'
                                    : 'bg-cyan/10 text-cyan-dark'
                                }`}
                              >
                                {field.type === 'datetime' && (
                                  <Clock className="w-3 h-3" />
                                )}
                                {field.type}
                              </span>
                            </td>
                            <td className="px-4 py-3.5 text-content-secondary hidden md:table-cell max-w-xs">
                              <p className="text-xs leading-relaxed">
                                {field.description}
                              </p>
                            </td>
                            <td className="px-4 py-3.5 hidden lg:table-cell">
                              <code className="text-xs font-mono text-content-secondary bg-surface px-2 py-1 rounded border border-border">
                                {field.example}
                              </code>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>

                {/* Sample CSV Visual */}
                <div className="bg-navy-900 rounded-xl p-4 overflow-x-auto">
                  <p className="text-content-muted text-[10px] uppercase tracking-widest font-semibold mb-3">
                    Sample CSV Format
                  </p>
                  <pre className="text-xs text-green-400 font-mono leading-relaxed whitespace-pre">
{`process_id,step_code,start_time,end_time,location
ORD-20240301-001,PICK,2024-03-01T08:00:00Z,2024-03-01T08:45:00Z,HAN-WH-01
ORD-20240301-001,PACK,2024-03-01T08:46:00Z,2024-03-01T09:10:00Z,HAN-WH-01
ORD-20240301-001,SHIP,2024-03-01T09:15:00Z,2024-03-01T11:30:00Z,HAN-PORT
ORD-20240301-002,PICK,2024-03-01T08:05:00Z,2024-03-01T09:50:00Z,HAN-WH-01`}
                  </pre>
                </div>
              </Section>

              {/* ─ Data Validation ─ */}
              <Section id="data-validation">
                <h3 className="text-lg font-semibold text-navy mb-4 mt-8 flex items-center gap-2">
                  <ShieldCheck className="w-4 h-4 text-navy" />
                  Data Validation
                </h3>
                <p className="text-content-secondary text-sm mb-4 leading-relaxed">
                  Vyn automatically validates your file on upload. Rows that
                  fail validation are{' '}
                  <strong className="text-danger">rejected</strong> and will not
                  contribute to the analysis. The following conditions cause
                  rejection:
                </p>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                  {[
                    {
                      label: 'Missing required columns',
                      desc: 'One or more of the 5 required columns is absent from the CSV header.',
                    },
                    {
                      label: 'Invalid timestamp format',
                      desc: 'start_time or end_time cannot be parsed as a valid ISO 8601 datetime.',
                    },
                    {
                      label: 'Negative duration',
                      desc: 'end_time is earlier than or equal to start_time for a given row.',
                    },
                    {
                      label: 'Empty critical fields',
                      desc: 'process_id, step_code, or location contains a null or blank value.',
                    },
                  ].map(({ label, desc }) => (
                    <div
                      key={label}
                      className="flex items-start gap-3 bg-danger-50 border border-danger/15 rounded-xl p-4"
                    >
                      <XCircle className="w-4 h-4 text-danger shrink-0 mt-0.5" />
                      <div>
                        <p className="text-sm font-semibold text-danger mb-0.5">
                          {label}
                        </p>
                        <p className="text-xs text-content-secondary leading-relaxed">
                          {desc}
                        </p>
                      </div>
                    </div>
                  ))}
                </div>
              </Section>
            </Section>

            {/* ═══════════════════════════════
                3. CORE LOGIC
                ═══════════════════════════════ */}
            <Section id="core-logic">
              <div className="flex items-center gap-3 mb-6">
                <div className="w-10 h-10 rounded-xl bg-orange/10 flex items-center justify-center shrink-0">
                  <Cpu className="w-5 h-5 text-orange" />
                </div>
                <div>
                  <h2 className="text-2xl font-bold text-navy">Core Logic</h2>
                  <p className="text-sm text-content-muted">
                    How step durations are evaluated
                  </p>
                </div>
              </div>

              <p className="text-content-secondary mb-8 leading-relaxed">
                Vyn evaluates each step by comparing its duration against a
                statistical baseline computed from the same step across all
                historical records. The engine is fully automated — there are
                no manually configured thresholds.
              </p>

              {/* ─ Baseline Statistics ─ */}
              <Section id="baseline-statistics">
                <h3 className="text-lg font-semibold text-navy mb-4 flex items-center gap-2">
                  <BarChart3 className="w-4 h-4 text-orange" />
                  Baseline Statistics
                </h3>
                <p className="text-sm text-content-secondary mb-5 leading-relaxed">
                  For every unique <code className="bg-surface border border-border px-1.5 py-0.5 rounded text-xs font-mono text-navy">step_code</code>, Vyn
                  computes three statistical measures from all historical
                  durations of that step:
                </p>

                <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 mb-6">
                  {[
                    {
                      stat: 'Mean (μ)',
                      icon: '—',
                      color: 'border-t-cyan',
                      textColor: 'text-cyan',
                      desc: 'The average duration for this step across all historical runs. Represents the typical expected time.',
                    },
                    {
                      stat: 'Std Dev (σ)',
                      icon: '±',
                      color: 'border-t-orange',
                      textColor: 'text-orange',
                      desc: 'How much individual step durations normally vary around the mean. A low σ means the step is very consistent.',
                    },
                    {
                      stat: '95th Percentile (p95)',
                      icon: '95%',
                      color: 'border-t-navy',
                      textColor: 'text-navy',
                      desc: 'The duration below which 95% of all executions fall. This is the primary ceiling for "normal" behaviour.',
                    },
                  ].map(({ stat, icon, color, textColor, desc }) => (
                    <div
                      key={stat}
                      className={`bg-white rounded-xl border border-border shadow-card p-5 border-t-4 ${color}`}
                    >
                      <div
                        className={`text-2xl font-black mb-2 ${textColor} font-mono`}
                      >
                        {icon}
                      </div>
                      <p className={`font-semibold text-sm mb-2 ${textColor}`}>
                        {stat}
                      </p>
                      <p className="text-xs text-content-muted leading-relaxed">
                        {desc}
                      </p>
                    </div>
                  ))}
                </div>

                <div className="bg-surface rounded-xl border border-border p-4">
                  <p className="text-xs text-content-muted font-semibold uppercase tracking-wider mb-2">
                    Normal Range Definition
                  </p>
                  <p className="text-sm text-content-secondary">
                    A step duration is considered{' '}
                    <strong className="text-success">normal</strong> if it falls
                    within:
                  </p>
                  <div className="mt-3 bg-white border border-border rounded-lg px-4 py-3 font-mono text-sm text-navy text-center">
                    [ μ − 2σ , p95 ]
                  </div>
                  <p className="text-xs text-content-muted mt-2 text-center">
                    The lower bound prevents flagging short steps; the upper
                    bound is the p95 ceiling.
                  </p>
                </div>
              </Section>

              {/* ─ Detection Rules ─ */}
              <Section id="detection-rules">
                <h3 className="text-lg font-semibold text-navy mb-4 mt-8 flex items-center gap-2">
                  <AlertTriangle className="w-4 h-4 text-orange" />
                  Bottleneck Detection Rules
                </h3>
                <p className="text-sm text-content-secondary mb-5 leading-relaxed">
                  An anomaly (bottleneck) is triggered if{' '}
                  <strong className="text-danger">either</strong> of the
                  following conditions is true for a given step execution:
                </p>

                <div className="space-y-4">
                  {/* Rule 1 */}
                  <div className="bg-white rounded-xl border border-border shadow-card overflow-hidden">
                    <div className="flex items-center gap-3 bg-danger/5 border-b border-danger/10 px-5 py-3">
                      <span className="w-6 h-6 rounded-full bg-danger/15 text-danger text-xs font-bold flex items-center justify-center">
                        1
                      </span>
                      <p className="font-semibold text-danger text-sm">
                        Duration exceeds the 95th Percentile
                      </p>
                    </div>
                    <div className="px-5 py-4">
                      <div className="font-mono text-sm text-navy bg-surface border border-border rounded-lg px-4 py-2.5 mb-3">
                        duration {'>'} p95
                      </div>
                      <p className="text-sm text-content-secondary leading-relaxed">
                        If this step took longer than 95% of all historical
                        executions, it is an outlier by definition. This is the
                        primary, most interpretable rule.
                      </p>
                    </div>
                  </div>

                  {/* Rule 2 */}
                  <div className="bg-white rounded-xl border border-border shadow-card overflow-hidden">
                    <div className="flex items-center gap-3 bg-danger/5 border-b border-danger/10 px-5 py-3">
                      <span className="w-6 h-6 rounded-full bg-danger/15 text-danger text-xs font-bold flex items-center justify-center">
                        2
                      </span>
                      <p className="font-semibold text-danger text-sm">
                        Z-score ≥ 2 (Statistical Significance)
                      </p>
                    </div>
                    <div className="px-5 py-4">
                      <div className="font-mono text-sm text-navy bg-surface border border-border rounded-lg px-4 py-2.5 mb-3">
                        z_score = (duration − μ) / σ ≥ 2
                      </div>
                      <p className="text-sm text-content-secondary leading-relaxed">
                        A z-score of 2 means the duration is at least 2
                        standard deviations above the mean — a statistically
                        significant deviation. This rule catches anomalies even
                        when p95 is very wide (high-variance steps).
                      </p>
                    </div>
                  </div>
                </div>

                <div className="flex items-start gap-3 bg-navy-50 border border-navy/10 rounded-xl p-4 mt-5">
                  <Info className="w-4 h-4 text-navy shrink-0 mt-0.5" />
                  <p className="text-sm text-content-secondary leading-relaxed">
                    <strong className="text-navy">Why two rules?</strong> The
                    p95 rule works best for consistent steps (low σ). The
                    z-score rule is better for variable steps where even a
                    single extreme execution stands out statistically. Using
                    both ensures robust detection across all step types.
                  </p>
                </div>
              </Section>
            </Section>

            {/* ═══════════════════════════════
                4. RISK SCORING
                ═══════════════════════════════ */}
            <Section id="risk-scoring">
              <div className="flex items-center gap-3 mb-6">
                <div className="w-10 h-10 rounded-xl bg-success/10 flex items-center justify-center shrink-0">
                  <ShieldCheck className="w-5 h-5 text-success" />
                </div>
                <div>
                  <h2 className="text-2xl font-bold text-navy">Risk Scoring</h2>
                  <p className="text-sm text-content-muted">
                    Quantifying how severe each bottleneck is
                  </p>
                </div>
              </div>

              <p className="text-content-secondary mb-6 leading-relaxed">
                Every step execution is assigned a{' '}
                <strong className="text-navy">risk percentage</strong> (0 – 100+)
                that tells you at a glance how far beyond normal its duration
                is. This score drives both the colour-coded indicators in the
                dashboard and the prioritisation of alerts.
              </p>

              {/* ─ Risk Formula ─ */}
              <Section id="risk-formula">
                <h3 className="text-lg font-semibold text-navy mb-4 flex items-center gap-2">
                  <Cpu className="w-4 h-4 text-navy" />
                  Scoring Formula
                </h3>

                <div className="bg-navy rounded-xl p-6 mb-6">
                  <p className="text-navy-100 text-xs uppercase tracking-widest font-semibold mb-3">
                    Risk Percent Formula
                  </p>
                  <div className="font-mono text-lg text-white text-center">
                    risk_percent = min(100, (duration / p95) × 100)
                  </div>
                  <p className="text-navy-100 text-xs text-center mt-3">
                    Capped at 100 for normal/warning ranges; scores above 100
                    indicate High Risk.
                  </p>
                </div>

                <p className="text-sm text-content-secondary leading-relaxed mb-6">
                  The formula expresses the duration as a percentage of the p95
                  baseline. A score of 100% means the step took exactly as long
                  as the 95th percentile. Scores above 100% indicate the step
                  exceeded the baseline and is an active bottleneck.
                </p>
              </Section>

              {/* ─ Risk Categories ─ */}
              <Section id="risk-categories">
                <h3 className="text-lg font-semibold text-navy mb-5 flex items-center gap-2">
                  <ShieldCheck className="w-4 h-4 text-navy" />
                  Risk Categories
                </h3>

                {/* Category Cards */}
                <div className="space-y-4 mb-8">
                  {riskCategories.map((cat) => (
                    <div
                      key={cat.level}
                      className={`rounded-xl border p-5 ${cat.bgClass} ${cat.borderClass}`}
                    >
                      <div className="flex items-center justify-between mb-2 flex-wrap gap-2">
                        <div className="flex items-center gap-2.5">
                          <span className={cat.colorClass}>{cat.icon}</span>
                          <span className={`font-bold text-base ${cat.colorClass}`}>
                            {cat.level}
                          </span>
                        </div>
                        <span
                          className={`text-xs font-semibold px-3 py-1 rounded-full ${cat.badgeClass}`}
                        >
                          {cat.threshold}
                        </span>
                      </div>
                      <p className="text-sm text-content-secondary leading-relaxed mb-2">
                        {cat.description}
                      </p>
                      <code className="text-xs font-mono text-content-muted bg-white/50 px-2 py-1 rounded border border-white/30">
                        Condition: {cat.condition}
                      </code>
                    </div>
                  ))}
                </div>

                {/* Risk Category Table — compact reference */}
                <h4 className="text-sm font-semibold text-navy mb-3">
                  Quick Reference Table
                </h4>
                <div className="bg-white rounded-xl border border-border shadow-card overflow-hidden">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="bg-surface border-b border-border">
                        <th className="text-left px-5 py-3 text-xs font-semibold uppercase tracking-wider text-content-muted">
                          Risk Level
                        </th>
                        <th className="text-left px-5 py-3 text-xs font-semibold uppercase tracking-wider text-content-muted">
                          Risk % Range
                        </th>
                        <th className="text-left px-5 py-3 text-xs font-semibold uppercase tracking-wider text-content-muted hidden sm:table-cell">
                          Recommended Action
                        </th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-border">
                      {[
                        {
                          level: 'Normal',
                          range: '< 80%',
                          action: 'No action required. Continue monitoring.',
                          color: 'text-success',
                          dot: 'bg-success',
                        },
                        {
                          level: 'Warning',
                          range: '80% – 100%',
                          action: 'Review step performance; consider proactive intervention.',
                          color: 'text-warning',
                          dot: 'bg-warning',
                        },
                        {
                          level: 'High Risk',
                          range: '> 100%',
                          action: 'Immediate investigation required. Active bottleneck.',
                          color: 'text-danger',
                          dot: 'bg-danger',
                        },
                      ].map(({ level, range, action, color, dot }) => (
                        <tr
                          key={level}
                          className="hover:bg-surface/50 transition-colors"
                        >
                          <td className="px-5 py-4">
                            <div className="flex items-center gap-2">
                              <span
                                className={`w-2.5 h-2.5 rounded-full ${dot} shrink-0`}
                              />
                              <span className={`font-semibold text-sm ${color}`}>
                                {level}
                              </span>
                            </div>
                          </td>
                          <td className="px-5 py-4">
                            <code
                              className={`font-mono text-xs font-semibold ${color}`}
                            >
                              {range}
                            </code>
                          </td>
                          <td className="px-5 py-4 text-content-secondary text-xs hidden sm:table-cell">
                            {action}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </Section>
            </Section>

          </div>
          {/* ─── End Main Content ─── */}
        </div>
      </div>

      {/* ── Scroll-to-top button ── */}
      {showScrollTop && (
        <button
          onClick={() => window.scrollTo({ top: 0, behavior: 'smooth' })}
          className="fixed bottom-6 right-6 w-10 h-10 bg-navy text-white rounded-xl shadow-elevated flex items-center justify-center hover:bg-navy-light transition-colors z-30 animate-fade-in"
          aria-label="Scroll to top"
        >
          <ArrowUp className="w-4 h-4" />
        </button>
      )}
    </div>
  );
};

export default DocumentationPage;
