import {
  Code2,
  ChevronRight,
  BookOpen,
  Upload,
  ArrowRight,
  Clock,
  Zap,
  Lock,
  Globe,
  Layers,
  CheckCircle,
} from 'lucide-react';

/* ──────────────────────────────────────────
   Placeholder endpoint cards (visual only)
   ────────────────────────────────────────── */

const placeholderEndpoints = [
  { method: 'POST', path: '/v2/datasets/upload', label: 'Upload a dataset' },
  { method: 'GET', path: '/v2/processes/{id}/report', label: 'Get bottleneck report' },
  { method: 'GET', path: '/v2/steps/{step_code}/baseline', label: 'Query step baseline' },
  { method: 'GET', path: '/v2/alerts', label: 'List active alerts' },
];

const methodColor: Record<string, string> = {
  GET: 'bg-success/10 text-success border-success/20',
  POST: 'bg-orange/10 text-orange border-orange/20',
  PUT: 'bg-cyan/10 text-cyan border-cyan/20',
  DELETE: 'bg-danger/10 text-danger border-danger/20',
};

/* ──────────────────────────────────────────
   ApiReference Page
   ────────────────────────────────────────── */

const ApiReferencePage = () => {
  return (
    <div className="min-h-screen bg-surface pt-20">

      {/* ── Page Header ── */}
      <div className="bg-white border-b border-border">
        <div className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 py-10">
          <div className="flex items-center gap-2 text-sm text-content-muted mb-3">
            <Code2 className="w-4 h-4" />
            <span>Resources</span>
            <ChevronRight className="w-3 h-3" />
            <span className="text-navy font-medium">API Reference</span>
          </div>
          <h1 className="text-3xl sm:text-4xl font-bold text-navy mb-2">
            API Reference
          </h1>
          <p className="text-content-secondary text-lg">
            Programmatic access to the Vyn Logistics Intelligence Platform.
          </p>
        </div>
      </div>

      {/* ── Main Coming-Soon Content ── */}
      <div className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 py-12">

        {/* Central hero card */}
        <div className="bg-white rounded-3xl border border-border shadow-elevated overflow-hidden">

          {/* Top accent strip */}
          <div className="h-1.5 w-full bg-gradient-to-r from-navy via-cyan to-orange" />

          <div className="px-8 py-14 text-center">
            {/* Icon badge */}
            <div className="relative inline-flex items-center justify-center mb-8">
              <div className="w-24 h-24 rounded-3xl bg-navy/5 border border-navy/10 flex items-center justify-center">
                <Code2 className="w-10 h-10 text-navy" />
              </div>
              {/* Coming Soon pill */}
              <span className="absolute -top-2 -right-2 inline-flex items-center gap-1 px-2.5 py-1 rounded-full bg-orange text-white text-[11px] font-bold shadow-elevated">
                <Clock className="w-3 h-3" />
                V2
              </span>
            </div>

            {/* Headline */}
            <h2 className="text-2xl sm:text-3xl font-bold text-navy mb-4">
              API Integration is Coming in V2
            </h2>

            {/* Message — MVP context accurate */}
            <p className="text-content-secondary text-base max-w-lg mx-auto leading-relaxed mb-3">
              The Vyn REST API is currently in development and will be released
              in <strong className="text-navy">Version 2</strong>. It will allow
              you to integrate bottleneck detection directly into your TMS, WMS,
              or custom pipelines — no manual CSV exports required.
            </p>
            <p className="text-content-secondary text-sm max-w-md mx-auto leading-relaxed mb-10">
              For the current version, please use the{' '}
              <strong className="text-orange">CSV Upload feature</strong> via
              the web interface. All MVP functionality — baseline statistics,
              anomaly detection, and risk scoring — is fully available today
              through the upload workflow.
            </p>

            {/* CTA buttons */}
            <div className="flex flex-col sm:flex-row items-center justify-center gap-3">
              <a
                href="/resources/docs"
                className="inline-flex items-center gap-2 px-6 py-3 rounded-xl bg-navy text-white text-sm font-semibold hover:bg-navy-light active:scale-[0.97] transition-all shadow-sm hover:shadow-md"
              >
                <BookOpen className="w-4 h-4" />
                Go to Documentation
                <ArrowRight className="w-4 h-4" />
              </a>
              <a
                href="/resources/help"
                className="inline-flex items-center gap-2 px-6 py-3 rounded-xl bg-surface border border-border text-content-secondary text-sm font-medium hover:border-border-dark hover:text-navy transition-all"
              >
                Visit Help Center
              </a>
            </div>
          </div>
        </div>

        {/* ── What to expect in V2 ── */}
        <div className="mt-12">
          <h3 className="text-lg font-bold text-navy mb-2 text-center">
            What's planned for the V2 API
          </h3>
          <p className="text-sm text-content-muted text-center mb-8">
            A preview of the capabilities coming in the next major release.
          </p>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            {[
              {
                icon: <Upload className="w-5 h-5 text-orange" />,
                bg: 'bg-orange/10',
                title: 'Programmatic Upload',
                desc: 'POST execution data without the web interface — ideal for automated pipelines.',
              },
              {
                icon: <Zap className="w-5 h-5 text-cyan" />,
                bg: 'bg-cyan/10',
                title: 'Real-time Results',
                desc: 'Get bottleneck reports via JSON response immediately after upload.',
              },
              {
                icon: <Lock className="w-5 h-5 text-navy" />,
                bg: 'bg-navy/10',
                title: 'API Key Auth',
                desc: 'Secure, scoped API keys with role-based access control per workspace.',
              },
              {
                icon: <Globe className="w-5 h-5 text-success" />,
                bg: 'bg-success/10',
                title: 'Webhook Alerts',
                desc: 'Push High Risk alerts to Slack, PagerDuty, or any HTTP endpoint.',
              },
            ].map(({ icon, bg, title, desc }) => (
              <div
                key={title}
                className="bg-white rounded-2xl border border-border shadow-card p-5"
              >
                <div
                  className={`w-10 h-10 rounded-xl ${bg} flex items-center justify-center mb-3`}
                >
                  {icon}
                </div>
                <p className="font-semibold text-navy text-sm mb-1">{title}</p>
                <p className="text-xs text-content-muted leading-relaxed">{desc}</p>
              </div>
            ))}
          </div>
        </div>

        {/* ── Blurred placeholder endpoint list ── */}
        <div className="mt-12 bg-white rounded-2xl border border-border shadow-card overflow-hidden">
          <div className="flex items-center justify-between px-5 py-4 border-b border-border">
            <div className="flex items-center gap-2">
              <Layers className="w-4 h-4 text-content-muted" />
              <span className="text-sm font-bold text-navy">
                V2 Endpoint Preview
              </span>
            </div>
            <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-[11px] font-semibold bg-warning/10 text-warning border border-warning/20">
              <Clock className="w-3 h-3" />
              Coming Soon
            </span>
          </div>

          {/* Blurred endpoint rows */}
          <div className="relative">
            <div className="divide-y divide-border pointer-events-none select-none blur-[3px] opacity-60">
              {placeholderEndpoints.map(({ method, path, label }) => (
                <div
                  key={path}
                  className="flex items-center gap-4 px-5 py-3.5"
                >
                  <span
                    className={`inline-block px-2 py-0.5 rounded text-[11px] font-bold border font-mono ${methodColor[method]}`}
                  >
                    {method}
                  </span>
                  <code className="font-mono text-xs text-navy font-semibold flex-1">
                    {path}
                  </code>
                  <span className="text-xs text-content-muted hidden sm:block">
                    {label}
                  </span>
                  <ChevronRight className="w-4 h-4 text-content-muted shrink-0" />
                </div>
              ))}
            </div>

            {/* Overlay */}
            <div className="absolute inset-0 flex items-center justify-center">
              <div className="bg-white/90 backdrop-blur-sm rounded-xl border border-border px-5 py-3 text-center shadow-elevated">
                <Clock className="w-5 h-5 text-orange mx-auto mb-1" />
                <p className="text-xs font-bold text-navy">
                  Available in V2
                </p>
              </div>
            </div>
          </div>
        </div>

        {/* ── Current alternative ── */}
        <div className="mt-8 bg-navy rounded-2xl p-8">
          <div className="flex flex-col sm:flex-row items-start sm:items-center gap-6">
            <div className="w-12 h-12 bg-white/10 rounded-xl flex items-center justify-center shrink-0">
              <Upload className="w-6 h-6 text-orange" />
            </div>
            <div className="flex-1">
              <h3 className="text-white font-bold text-lg mb-1">
                Use CSV Upload Today
              </h3>
              <p className="text-navy-100 text-sm leading-relaxed max-w-xl">
                Everything the V2 API will do is available now via the web
                interface. Prepare your data in the{' '}
                <strong className="text-white">5-column CSV schema</strong>,
                upload it, and get a full bottleneck analysis — including
                baseline statistics, anomaly flags, and risk scores.
              </p>
              <ul className="mt-3 space-y-1">
                {[
                  'No code or API keys required',
                  'Full statistical analysis in seconds',
                  'Risk scores and severity badges for every step',
                ].map((item) => (
                  <li
                    key={item}
                    className="flex items-center gap-2 text-xs text-navy-100"
                  >
                    <CheckCircle className="w-3.5 h-3.5 text-success shrink-0" />
                    {item}
                  </li>
                ))}
              </ul>
            </div>
            <a
              href="/resources/docs"
              className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl bg-orange hover:bg-orange-dark text-white text-sm font-bold transition-colors shadow-sm shrink-0"
            >
              <BookOpen className="w-4 h-4" />
              Read the Docs
              <ArrowRight className="w-4 h-4" />
            </a>
          </div>
        </div>

      </div>
    </div>
  );
};

export default ApiReferencePage;
