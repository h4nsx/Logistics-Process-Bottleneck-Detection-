import { useState, useMemo } from 'react';
import {
  HelpCircle,
  Search,
  ChevronDown,
  ChevronRight,
  Upload,
  Cpu,
  ShieldCheck,
  Database,
  AlertTriangle,
  CheckCircle,
  XCircle,
  BookOpen,
  MessageSquare,
  Mail,
  Clock,
  FileText,
  BarChart3,
  X,
} from 'lucide-react';

/* ──────────────────────────────────────────
   Types
   ────────────────────────────────────────── */

interface FAQ {
  id: string;
  question: string;
  answer: React.ReactNode;
  category: string;
  tags: string[];
}

/* ──────────────────────────────────────────
   FAQ Data — MVP-accurate content
   ────────────────────────────────────────── */

const faqs: FAQ[] = [
  /* ── Upload & Data ── */
  {
    id: 'csv-upload-failing',
    category: 'Upload & Data',
    tags: ['csv', 'upload', 'error', 'validation'],
    question: 'Why is my CSV upload failing?',
    answer: (
      <div className="space-y-3 text-sm text-content-secondary leading-relaxed">
        <p>
          Upload failures are almost always caused by a{' '}
          <strong className="text-navy">data validation error</strong>. Vyn
          strictly validates every row before processing. Your upload will be
          rejected if any of the following conditions are found:
        </p>
        <ul className="space-y-2 mt-2">
          {[
            {
              icon: <XCircle className="w-4 h-4 text-danger shrink-0 mt-0.5" />,
              text: (
                <>
                  <strong className="text-danger">Missing required columns</strong> — Your CSV header must contain
                  exactly these five columns (case-sensitive):{' '}
                  <code className="bg-surface border border-border px-1.5 py-0.5 rounded text-xs font-mono text-navy">
                    process_id
                  </code>,{' '}
                  <code className="bg-surface border border-border px-1.5 py-0.5 rounded text-xs font-mono text-navy">
                    step_code
                  </code>,{' '}
                  <code className="bg-surface border border-border px-1.5 py-0.5 rounded text-xs font-mono text-navy">
                    start_time
                  </code>,{' '}
                  <code className="bg-surface border border-border px-1.5 py-0.5 rounded text-xs font-mono text-navy">
                    end_time
                  </code>,{' '}
                  <code className="bg-surface border border-border px-1.5 py-0.5 rounded text-xs font-mono text-navy">
                    location
                  </code>.
                </>
              ),
            },
            {
              icon: <XCircle className="w-4 h-4 text-danger shrink-0 mt-0.5" />,
              text: (
                <>
                  <strong className="text-danger">Invalid timestamp format</strong> — Timestamps must be valid{' '}
                  <strong>ISO 8601</strong> datetimes (e.g.,{' '}
                  <code className="bg-surface border border-border px-1.5 py-0.5 rounded text-xs font-mono text-navy">
                    2024-03-01T08:00:00Z
                  </code>
                  ). Plain date strings like <em>"01/03/2024 08:00"</em> will fail.
                </>
              ),
            },
            {
              icon: <XCircle className="w-4 h-4 text-danger shrink-0 mt-0.5" />,
              text: (
                <>
                  <strong className="text-danger">end_time is before start_time</strong> — If a row has
                  an <code className="bg-surface border border-border px-1.5 py-0.5 rounded text-xs font-mono text-navy">end_time</code>{' '}
                  earlier than or equal to its{' '}
                  <code className="bg-surface border border-border px-1.5 py-0.5 rounded text-xs font-mono text-navy">
                    start_time
                  </code>
                  , the row is invalid and rejected. This is the most common cause of upload failures.
                </>
              ),
            },
            {
              icon: <XCircle className="w-4 h-4 text-danger shrink-0 mt-0.5" />,
              text: (
                <>
                  <strong className="text-danger">Empty critical fields</strong> —{' '}
                  <code className="bg-surface border border-border px-1.5 py-0.5 rounded text-xs font-mono text-navy">
                    process_id
                  </code>
                  ,{' '}
                  <code className="bg-surface border border-border px-1.5 py-0.5 rounded text-xs font-mono text-navy">
                    step_code
                  </code>
                  , or{' '}
                  <code className="bg-surface border border-border px-1.5 py-0.5 rounded text-xs font-mono text-navy">
                    location
                  </code>{' '}
                  cannot be blank or null.
                </>
              ),
            },
          ].map(({ icon, text }, i) => (
            <li key={i} className="flex items-start gap-2.5">
              {icon}
              <span>{text}</span>
            </li>
          ))}
        </ul>
        <div className="bg-orange-50 border border-orange/20 rounded-lg p-3 mt-3 flex items-start gap-2.5">
          <AlertTriangle className="w-4 h-4 text-orange shrink-0 mt-0.5" />
          <p className="text-xs text-content-secondary">
            <strong className="text-orange">Tip:</strong> Open your CSV in a
            spreadsheet editor and check the header row first. Column names are
            case-sensitive — <code className="font-mono">Start_Time</code> will
            fail; <code className="font-mono">start_time</code> is correct.
          </p>
        </div>
      </div>
    ),
  },
  {
    id: 'required-columns',
    category: 'Upload & Data',
    tags: ['csv', 'schema', 'columns', 'format'],
    question: 'What are the required columns and their exact format?',
    answer: (
      <div className="space-y-3 text-sm text-content-secondary leading-relaxed">
        <p>Your CSV must contain exactly these five columns:</p>
        <div className="overflow-x-auto rounded-lg border border-border">
          <table className="w-full text-xs">
            <thead>
              <tr className="bg-navy text-white">
                <th className="text-left px-4 py-2.5 font-semibold">Column</th>
                <th className="text-left px-4 py-2.5 font-semibold">Type</th>
                <th className="text-left px-4 py-2.5 font-semibold hidden sm:table-cell">Example</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border">
              {[
                { col: 'process_id', type: 'string', ex: 'ORD-20240301-001' },
                { col: 'step_code', type: 'string', ex: 'PICK' },
                { col: 'start_time', type: 'ISO 8601 datetime', ex: '2024-03-01T08:00:00Z' },
                { col: 'end_time', type: 'ISO 8601 datetime', ex: '2024-03-01T08:45:00Z' },
                { col: 'location', type: 'string', ex: 'HAN-WH-01' },
              ].map(({ col, type, ex }) => (
                <tr key={col} className="hover:bg-surface/50">
                  <td className="px-4 py-2.5 font-mono font-semibold text-navy">{col}</td>
                  <td className="px-4 py-2.5 text-content-muted">{type}</td>
                  <td className="px-4 py-2.5 font-mono text-content-secondary hidden sm:table-cell">{ex}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="text-xs text-content-muted">
          Column names are <strong>case-sensitive</strong>. Always use lowercase
          with underscores exactly as shown above.
        </p>
      </div>
    ),
  },
  {
    id: 'how-to-upload',
    category: 'Upload & Data',
    tags: ['upload', 'csv', 'get started'],
    question: 'How do I upload my data?',
    answer: (
      <div className="space-y-3 text-sm text-content-secondary leading-relaxed">
        <p>
          Vyn accepts data exclusively through{' '}
          <strong className="text-navy">CSV file upload</strong>. There is no
          direct API endpoint or database connector in the current version. To
          get started:
        </p>
        <ol className="space-y-2 list-none">
          {[
            'Prepare your CSV file with the required 5-column schema.',
            'Navigate to the Upload section from the main dashboard.',
            'Drag-and-drop your CSV file or click to browse.',
            'Vyn validates your file automatically and reports any row-level errors.',
            'Once validated, the analysis runs and results appear in your dashboard.',
          ].map((step, i) => (
            <li key={i} className="flex items-start gap-2.5">
              <span className="w-5 h-5 rounded-full bg-cyan/15 text-cyan text-[11px] font-bold flex items-center justify-center shrink-0 mt-0.5">
                {i + 1}
              </span>
              <span>{step}</span>
            </li>
          ))}
        </ol>
      </div>
    ),
  },

  /* ── AI & Detection ── */
  {
    id: 'predict-future-delays',
    category: 'AI & Detection',
    tags: ['ai', 'prediction', 'future', 'forecast'],
    question: 'Does Vyn predict future delays?',
    answer: (
      <div className="space-y-3 text-sm text-content-secondary leading-relaxed">
        <div className="flex items-start gap-3 bg-navy/5 border border-navy/10 rounded-xl p-4">
          <XCircle className="w-5 h-5 text-danger shrink-0 mt-0.5" />
          <div>
            <p className="font-semibold text-navy mb-1">
              No — Vyn does not predict future delays.
            </p>
            <p className="text-sm text-content-secondary">
              Vyn is a <strong>statistical intelligence</strong> system. It
              analyses <em>historical</em> execution data to detect which past
              step executions were abnormally slow. It does not use machine
              learning models to forecast what will happen in the future.
            </p>
          </div>
        </div>
        <p>
          The system computes a statistical baseline (mean, standard deviation,
          95th percentile) from your historical records. Any past execution that
          significantly exceeded this baseline is flagged as an anomaly and
          assigned a risk score.
        </p>
        <p>
          Think of Vyn as a{' '}
          <strong className="text-navy">
            diagnostic tool, not a forecasting tool
          </strong>
          . It tells you <em>"this step has been a bottleneck in the past"</em>,
          which empowers your team to make informed operational decisions going
          forward.
        </p>
      </div>
    ),
  },
  {
    id: 'auto-optimize',
    category: 'AI & Detection',
    tags: ['optimize', 'automate', 'ai', 'actions'],
    question: 'Does Vyn automatically optimise my supply chain?',
    answer: (
      <div className="space-y-3 text-sm text-content-secondary leading-relaxed">
        <div className="flex items-start gap-3 bg-navy/5 border border-navy/10 rounded-xl p-4">
          <XCircle className="w-5 h-5 text-danger shrink-0 mt-0.5" />
          <div>
            <p className="font-semibold text-navy mb-1">
              No — Vyn does not take automatic actions.
            </p>
            <p className="text-sm text-content-secondary">
              Vyn is a <strong>decision-support system</strong>, not an
              automation engine. It surfaces insights and risk scores so your
              operations team can make better, data-driven decisions — but it
              never directly modifies routes, schedules, or any operational
              systems.
            </p>
          </div>
        </div>
        <p>What Vyn does do:</p>
        <ul className="space-y-1.5">
          {[
            'Identifies which process steps are statistically abnormal.',
            'Assigns a risk score (Normal / Warning / High Risk) to each step.',
            'Highlights the most severe bottlenecks for your team to prioritize.',
          ].map((item, i) => (
            <li key={i} className="flex items-start gap-2">
              <CheckCircle className="w-4 h-4 text-success shrink-0 mt-0.5" />
              <span>{item}</span>
            </li>
          ))}
        </ul>
        <p>What Vyn does NOT do:</p>
        <ul className="space-y-1.5">
          {[
            'Automatically re-route shipments or adjust schedules.',
            'Provide causal analysis (it shows the "what", not the "why").',
            'Connect to or modify external operational systems.',
            'Make predictions or recommendations about future operations.',
          ].map((item, i) => (
            <li key={i} className="flex items-start gap-2">
              <XCircle className="w-4 h-4 text-danger shrink-0 mt-0.5" />
              <span>{item}</span>
            </li>
          ))}
        </ul>
      </div>
    ),
  },
  {
    id: 'how-anomaly-detected',
    category: 'AI & Detection',
    tags: ['anomaly', 'detection', 'z-score', 'p95', 'logic'],
    question: 'How does Vyn decide if a step is a bottleneck?',
    answer: (
      <div className="space-y-3 text-sm text-content-secondary leading-relaxed">
        <p>
          A step execution is flagged as a bottleneck (anomaly) if{' '}
          <strong className="text-danger">either</strong> of two statistical
          rules is triggered:
        </p>
        <div className="space-y-3">
          {[
            {
              rule: 'Rule 1',
              title: 'Duration exceeds p95',
              formula: 'duration > p95',
              desc: "The step took longer than 95% of all historical executions of the same step. This makes it a statistical outlier by definition.",
              color: 'border-t-danger',
            },
            {
              rule: 'Rule 2',
              title: 'High Z-score',
              formula: 'z_score = (duration − mean) / std ≥ 2',
              desc: "The step duration is 2+ standard deviations above the mean — statistically significant even for high-variance steps.",
              color: 'border-t-orange',
            },
          ].map(({ rule, title, formula, desc, color }) => (
            <div
              key={rule}
              className={`bg-white rounded-xl border border-border p-4 border-t-4 ${color}`}
            >
              <div className="flex items-center gap-2 mb-1.5">
                <span className="text-xs font-bold text-content-muted">{rule}:</span>
                <span className="text-sm font-semibold text-navy">{title}</span>
              </div>
              <code className="block text-xs font-mono text-navy bg-surface border border-border rounded px-3 py-2 mb-2">
                {formula}
              </code>
              <p className="text-xs text-content-muted leading-relaxed">{desc}</p>
            </div>
          ))}
        </div>
      </div>
    ),
  },
  {
    id: 'causal-analysis',
    category: 'AI & Detection',
    tags: ['cause', 'why', 'root cause', 'analysis'],
    question: 'Can Vyn tell me why a bottleneck occurred?',
    answer: (
      <div className="space-y-3 text-sm text-content-secondary leading-relaxed">
        <div className="flex items-start gap-3 bg-warning-50 border border-warning/20 rounded-xl p-4">
          <AlertTriangle className="w-5 h-5 text-warning shrink-0 mt-0.5" />
          <p>
            <strong className="text-warning">Not in the current version.</strong>{' '}
            Vyn identifies <em>that</em> a bottleneck occurred and{' '}
            <em>how severe</em> it was — but it does not perform causal analysis
            to explain <em>why</em> it happened.
          </p>
        </div>
        <p>
          The system is built on statistical anomaly detection: it compares step
          durations against historical baselines derived from your own data. The
          "why" — driver shortages, equipment failures, peak demand — requires
          contextual knowledge that lives outside the dataset.
        </p>
        <p>
          Use Vyn's risk scores and bottleneck reports to{' '}
          <strong className="text-navy">direct your team's investigation</strong>{' '}
          to the right steps, locations, and time windows. The human expert then
          applies domain knowledge to identify the root cause.
        </p>
      </div>
    ),
  },

  /* ── Risk Scoring ── */
  {
    id: 'risk-score-meaning',
    category: 'Risk Scoring',
    tags: ['risk', 'score', 'percentage', 'meaning'],
    question: 'What does the risk percentage score mean?',
    answer: (
      <div className="space-y-3 text-sm text-content-secondary leading-relaxed">
        <p>
          The{' '}
          <strong className="text-navy">risk percentage</strong> tells you how a
          step's duration compares to its 95th percentile (p95) baseline:
        </p>
        <div className="bg-navy rounded-xl p-4 text-center font-mono text-white text-sm">
          risk_percent = min(100, (duration / p95) × 100)
        </div>
        <p>The score maps to three categories:</p>
        <div className="space-y-2">
          {[
            {
              level: 'Normal',
              range: '< 80%',
              desc: 'Well within historical norms. No action needed.',
              icon: <CheckCircle className="w-4 h-4" />,
              color: 'text-success',
              bg: 'bg-success-50 border-success/20',
            },
            {
              level: 'Warning',
              range: '80% – 100%',
              desc: 'Approaching the baseline ceiling. Monitor closely.',
              icon: <AlertTriangle className="w-4 h-4" />,
              color: 'text-warning',
              bg: 'bg-warning-50 border-warning/20',
            },
            {
              level: 'High Risk',
              range: '> 100%',
              desc: 'Exceeded p95 baseline. Active bottleneck — investigate immediately.',
              icon: <XCircle className="w-4 h-4" />,
              color: 'text-danger',
              bg: 'bg-danger-50 border-danger/20',
            },
          ].map(({ level, range, desc, icon, color, bg }) => (
            <div
              key={level}
              className={`flex items-center gap-3 rounded-lg border p-3 ${bg}`}
            >
              <span className={color}>{icon}</span>
              <div className="flex-1 min-w-0">
                <span className={`font-semibold text-sm ${color}`}>{level}</span>
                <span className="text-content-muted text-xs ml-2">({range})</span>
                <p className="text-xs text-content-secondary mt-0.5">{desc}</p>
              </div>
            </div>
          ))}
        </div>
      </div>
    ),
  },
  {
    id: 'high-risk-all-time',
    category: 'Risk Scoring',
    tags: ['risk', 'high risk', 'normal', 'always'],
    question: 'Why does a step always appear as High Risk?',
    answer: (
      <div className="space-y-3 text-sm text-content-secondary leading-relaxed">
        <p>
          If a particular step consistently appears as High Risk, it usually
          means one of the following:
        </p>
        <ul className="space-y-2">
          {[
            'The step genuinely has a systemic delay that needs operational attention.',
            'Your dataset is too small for that step — with few historical records, the baseline (mean and p95) may be skewed by a handful of slow outliers.',
            'The step has highly variable durations across locations or seasons. Consider filtering your dataset to a specific location or time window for a more accurate baseline.',
          ].map((item, i) => (
            <li key={i} className="flex items-start gap-2">
              <ChevronRight className="w-4 h-4 text-orange shrink-0 mt-0.5" />
              <span className="text-sm">{item}</span>
            </li>
          ))}
        </ul>
        <div className="bg-cyan/5 border border-cyan/15 rounded-lg p-3">
          <p className="text-xs text-content-secondary">
            <strong className="text-cyan">Recommendation:</strong> Aim for at
            least <strong>30+ historical records</strong> per step_code to get a
            stable statistical baseline.
          </p>
        </div>
      </div>
    ),
  },

  /* ── Getting Started ── */
  {
    id: 'minimum-data-size',
    category: 'Getting Started',
    tags: ['data', 'minimum', 'records', 'sample size'],
    question: 'How much historical data do I need?',
    answer: (
      <div className="space-y-3 text-sm text-content-secondary leading-relaxed">
        <p>
          The quality of Vyn's analysis depends directly on the amount of
          historical data you provide. The more records per step, the more
          statistically reliable the baseline.
        </p>
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
          {[
            { label: 'Minimum', count: '10+', desc: 'Functional but less reliable baselines.', color: 'text-warning', bg: 'bg-warning-50 border-warning/20' },
            { label: 'Recommended', count: '30+', desc: 'Solid baselines for most logistics steps.', color: 'text-cyan', bg: 'bg-cyan/5 border-cyan/20' },
            { label: 'Optimal', count: '100+', desc: 'Highly accurate statistical detection.', color: 'text-success', bg: 'bg-success-50 border-success/20' },
          ].map(({ label, count, desc, color, bg }) => (
            <div key={label} className={`rounded-xl border p-4 text-center ${bg}`}>
              <p className={`text-2xl font-black ${color} font-mono`}>{count}</p>
              <p className={`text-xs font-semibold ${color} mb-1`}>{label}</p>
              <p className="text-xs text-content-muted leading-relaxed">{desc}</p>
            </div>
          ))}
        </div>
        <p className="text-xs text-content-muted">
          These counts are <em>per unique step_code</em>, not the total rows in
          your CSV.
        </p>
      </div>
    ),
  },
  {
    id: 'external-integrations',
    category: 'Getting Started',
    tags: ['api', 'integration', 'external', 'connect'],
    question: 'Can I connect Vyn to my TMS or WMS via API?',
    answer: (
      <div className="space-y-3 text-sm text-content-secondary leading-relaxed">
        <div className="flex items-start gap-3 bg-navy/5 border border-navy/10 rounded-xl p-4">
          <AlertTriangle className="w-5 h-5 text-warning shrink-0 mt-0.5" />
          <p>
            <strong className="text-navy">
              Not in the current version.
            </strong>{' '}
            Vyn currently supports{' '}
            <strong>CSV file upload only</strong>. There are no external API
            integrations, database connectors, or live data feeds at this stage.
          </p>
        </div>
        <p>
          To use Vyn, export your process execution data from your TMS, WMS, or
          ERP as a CSV file, format it to the required schema, and upload it
          through the platform. API integrations are planned for future releases.
        </p>
      </div>
    ),
  },
];

/* ──────────────────────────────────────────
   Category Config
   ────────────────────────────────────────── */

const categories = [
  { label: 'All Topics', value: 'all', icon: <HelpCircle className="w-4 h-4" /> },
  { label: 'Upload & Data', value: 'Upload & Data', icon: <Upload className="w-4 h-4" /> },
  { label: 'AI & Detection', value: 'AI & Detection', icon: <Cpu className="w-4 h-4" /> },
  { label: 'Risk Scoring', value: 'Risk Scoring', icon: <ShieldCheck className="w-4 h-4" /> },
  { label: 'Getting Started', value: 'Getting Started', icon: <BookOpen className="w-4 h-4" /> },
];

/* ──────────────────────────────────────────
   Accordion Item
   ────────────────────────────────────────── */

const AccordionItem = ({
  faq,
  isOpen,
  onToggle,
}: {
  faq: FAQ;
  isOpen: boolean;
  onToggle: () => void;
}) => (
  <div
    className={`rounded-xl border transition-all duration-200 ${
      isOpen
        ? 'border-navy/20 shadow-elevated bg-white'
        : 'border-border bg-white hover:border-border-dark hover:shadow-card'
    }`}
  >
    <button
      onClick={onToggle}
      className="w-full flex items-start gap-4 px-5 py-4 text-left focus:outline-none focus-visible:ring-2 focus-visible:ring-orange rounded-xl"
      aria-expanded={isOpen}
    >
      {/* Question icon */}
      <span
        className={`w-7 h-7 rounded-lg flex items-center justify-center shrink-0 mt-0.5 transition-colors duration-200 ${
          isOpen ? 'bg-orange/15 text-orange' : 'bg-surface text-content-muted'
        }`}
      >
        <HelpCircle className="w-4 h-4" />
      </span>

      {/* Question text */}
      <div className="flex-1 min-w-0">
        <p
          className={`font-semibold text-sm sm:text-[15px] leading-snug transition-colors duration-150 ${
            isOpen ? 'text-navy' : 'text-content-primary'
          }`}
        >
          {faq.question}
        </p>
        <span
          className={`inline-block mt-1.5 text-[11px] font-medium px-2 py-0.5 rounded-full ${
            faq.category === 'Upload & Data'
              ? 'bg-cyan/10 text-cyan-dark'
              : faq.category === 'AI & Detection'
              ? 'bg-orange/10 text-orange'
              : faq.category === 'Risk Scoring'
              ? 'bg-success/10 text-success'
              : 'bg-navy/10 text-navy'
          }`}
        >
          {faq.category}
        </span>
      </div>

      {/* Chevron */}
      <ChevronDown
        className={`w-5 h-5 shrink-0 mt-0.5 text-content-muted transition-transform duration-300 ${
          isOpen ? 'rotate-180 text-orange' : ''
        }`}
      />
    </button>

    {/* Answer */}
    {isOpen && (
      <div className="px-5 pb-5 pt-1 border-t border-border/60 mt-0 animate-fade-in">
        <div className="pl-11">{faq.answer}</div>
      </div>
    )}
  </div>
);

/* ──────────────────────────────────────────
   Main Help Center Page
   ────────────────────────────────────────── */

const HelpCenterPage = () => {
  const [query, setQuery] = useState('');
  const [activeCategory, setActiveCategory] = useState('all');
  const [openId, setOpenId] = useState<string | null>('csv-upload-failing');

  const filtered = useMemo(() => {
    let result = faqs;

    if (activeCategory !== 'all') {
      result = result.filter((f) => f.category === activeCategory);
    }

    if (query.trim()) {
      const q = query.toLowerCase();
      result = result.filter(
        (f) =>
          f.question.toLowerCase().includes(q) ||
          f.tags.some((t) => t.includes(q)) ||
          f.category.toLowerCase().includes(q)
      );
    }

    return result;
  }, [query, activeCategory]);

  const toggle = (id: string) => setOpenId(openId === id ? null : id);

  return (
    <div className="min-h-screen bg-surface pt-20">

      {/* ── Hero Header ── */}
      <div className="bg-white border-b border-border">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-12 text-center">
          {/* Breadcrumb */}
          <div className="flex items-center justify-center gap-2 text-sm text-content-muted mb-4">
            <HelpCircle className="w-4 h-4" />
            <span>Resources</span>
            <ChevronRight className="w-3 h-3" />
            <span className="text-navy font-medium">Help Center</span>
          </div>

          <h1 className="text-3xl sm:text-4xl font-bold text-navy mb-3">
            Help Center
          </h1>
          <p className="text-content-secondary text-lg mb-8 max-w-xl mx-auto">
            Find answers to common questions about uploading your data,
            understanding detections, and interpreting risk scores.
          </p>

          {/* ── Search Bar ── */}
          <div className="relative max-w-xl mx-auto">
            <Search className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-content-muted pointer-events-none" />
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Search questions, e.g. 'upload', 'prediction', 'csv'..."
              className="w-full pl-12 pr-10 py-3.5 rounded-xl border border-border bg-surface text-sm text-content-primary placeholder-content-muted focus:outline-none focus:ring-2 focus:ring-orange/40 focus:border-orange transition-all shadow-card"
            />
            {query && (
              <button
                onClick={() => setQuery('')}
                className="absolute right-4 top-1/2 -translate-y-1/2 text-content-muted hover:text-content-secondary transition-colors"
                aria-label="Clear search"
              >
                <X className="w-4 h-4" />
              </button>
            )}
          </div>

          {/* Stats pills */}
          <div className="flex flex-wrap items-center justify-center gap-3 mt-6">
            {[
              { icon: <FileText className="w-3.5 h-3.5" />, label: `${faqs.length} Articles` },
              { icon: <Database className="w-3.5 h-3.5" />, label: '4 Categories' },
              { icon: <Clock className="w-3.5 h-3.5" />, label: '2 min avg. read' },
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

      {/* ── Body ── */}
      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-10">

        {/* ── Category Tabs ── */}
        <div className="flex flex-wrap gap-2 mb-8">
          {categories.map((cat) => (
            <button
              key={cat.value}
              onClick={() => {
                setActiveCategory(cat.value);
                setOpenId(null);
              }}
              className={`inline-flex items-center gap-1.5 px-4 py-2 rounded-lg text-sm font-medium transition-all duration-150 ${
                activeCategory === cat.value
                  ? 'bg-navy text-white shadow-elevated'
                  : 'bg-white text-content-secondary border border-border hover:border-border-dark hover:text-navy'
              }`}
            >
              <span
                className={
                  activeCategory === cat.value
                    ? 'text-orange'
                    : 'text-content-muted'
                }
              >
                {cat.icon}
              </span>
              {cat.label}
              <span
                className={`ml-1 text-[11px] font-bold px-1.5 py-0.5 rounded-full ${
                  activeCategory === cat.value
                    ? 'bg-white/20 text-white'
                    : 'bg-surface text-content-muted'
                }`}
              >
                {cat.value === 'all'
                  ? faqs.length
                  : faqs.filter((f) => f.category === cat.value).length}
              </span>
            </button>
          ))}
        </div>

        {/* ── Results count / empty state ── */}
        {query && (
          <p className="text-sm text-content-muted mb-5">
            {filtered.length > 0 ? (
              <>
                Found <strong className="text-navy">{filtered.length}</strong>{' '}
                result{filtered.length !== 1 ? 's' : ''} for "
                <em>{query}</em>"
              </>
            ) : null}
          </p>
        )}

        {/* ── FAQ Accordion ── */}
        {filtered.length > 0 ? (
          <div className="space-y-3">
            {filtered.map((faq) => (
              <AccordionItem
                key={faq.id}
                faq={faq}
                isOpen={openId === faq.id}
                onToggle={() => toggle(faq.id)}
              />
            ))}
          </div>
        ) : (
          /* Empty state */
          <div className="text-center py-16">
            <div className="w-16 h-16 bg-surface rounded-2xl flex items-center justify-center mx-auto mb-4 border border-border">
              <Search className="w-7 h-7 text-content-muted" />
            </div>
            <h3 className="text-lg font-semibold text-navy mb-2">
              No results found
            </h3>
            <p className="text-content-secondary text-sm mb-6 max-w-xs mx-auto">
              We couldn't find any articles matching "
              <strong>{query}</strong>". Try a different keyword or browse by
              category.
            </p>
            <button
              onClick={() => {
                setQuery('');
                setActiveCategory('all');
              }}
              className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-orange text-white text-sm font-semibold hover:bg-orange-dark transition-colors"
            >
              <X className="w-4 h-4" />
              Clear Search
            </button>
          </div>
        )}

        {/* ── Still need help? ── */}
        <div className="mt-14 bg-navy rounded-2xl p-8 text-center">
          <div className="w-12 h-12 bg-white/10 rounded-xl flex items-center justify-center mx-auto mb-4">
            <MessageSquare className="w-6 h-6 text-orange" />
          </div>
          <h3 className="text-xl font-bold text-white mb-2">
            Still have questions?
          </h3>
          <p className="text-navy-100 text-sm mb-6 max-w-sm mx-auto">
            Can't find what you're looking for? Our team is here to help you get
            the most out of Vyn.
          </p>
          <div className="flex flex-col sm:flex-row items-center justify-center gap-3">
            <a
              href="/resources/docs"
              className="inline-flex items-center gap-2 px-5 py-2.5 rounded-lg bg-white/10 hover:bg-white/20 text-white text-sm font-semibold transition-colors border border-white/20"
            >
              <BookOpen className="w-4 h-4" />
              Read the Docs
            </a>
            <a
              href="/about-us/contact"
              className="inline-flex items-center gap-2 px-5 py-2.5 rounded-lg bg-orange hover:bg-orange-dark text-white text-sm font-semibold transition-colors shadow-sm hover:shadow-md"
            >
              <Mail className="w-4 h-4" />
              Contact Support
            </a>
          </div>
        </div>

        {/* ── Topic overview grid ── */}
        <div className="mt-10 grid grid-cols-1 sm:grid-cols-2 gap-4">
          {categories.slice(1).map((cat) => {
            const count = faqs.filter((f) => f.category === cat.value).length;
            const catIcon =
              cat.value === 'Upload & Data'
                ? <Upload className="w-5 h-5 text-cyan" />
                : cat.value === 'AI & Detection'
                ? <Cpu className="w-5 h-5 text-orange" />
                : cat.value === 'Risk Scoring'
                ? <BarChart3 className="w-5 h-5 text-success" />
                : <BookOpen className="w-5 h-5 text-navy" />;
            return (
              <button
                key={cat.value}
                onClick={() => {
                  setActiveCategory(cat.value);
                  setOpenId(null);
                  window.scrollTo({ top: 0, behavior: 'smooth' });
                }}
                className="flex items-center gap-4 bg-white rounded-xl border border-border p-4 hover:border-border-dark hover:shadow-card transition-all text-left group"
              >
                <div className="w-10 h-10 rounded-xl bg-surface flex items-center justify-center shrink-0 group-hover:scale-110 transition-transform">
                  {catIcon}
                </div>
                <div>
                  <p className="font-semibold text-navy text-sm">{cat.label}</p>
                  <p className="text-xs text-content-muted">{count} articles</p>
                </div>
                <ChevronRight className="w-4 h-4 text-content-muted ml-auto group-hover:text-orange transition-colors" />
              </button>
            );
          })}
        </div>
      </div>
    </div>
  );
};

export default HelpCenterPage;
