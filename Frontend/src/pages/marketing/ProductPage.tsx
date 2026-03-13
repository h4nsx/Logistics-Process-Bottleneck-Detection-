import { Link } from 'react-router-dom';
import {
  ArrowRight,
  Layers,
  Settings,
  Cpu,
  Target,
  BarChart3,
  Zap,
  Upload,
  GitBranch,
  TrendingUp,
  CheckCircle2,
  Clock,
  Shield,
  Sparkles,
  Columns3,
} from 'lucide-react';

/* ──────────────────────────────────────────
   Section 1: Hero
   ────────────────────────────────────────── */

const HeroSection = () => (
  <section className="relative overflow-hidden bg-gradient-to-b from-navy-50 via-white to-white py-20 lg:py-28">
    {/* Background decorations */}
    <div className="absolute top-0 left-1/2 -translate-x-1/2 w-[800px] h-[400px] bg-gradient-to-b from-orange/5 to-transparent rounded-full blur-3xl animate-pulse" style={{ animationDuration: '4s' }} />
    <div className="absolute -top-20 -right-20 w-72 h-72 bg-cyan/5 rounded-full blur-3xl animate-float" />
    <div className="absolute -bottom-10 -left-20 w-64 h-64 bg-orange/5 rounded-full blur-3xl animate-float-delayed" />

    <div className="absolute inset-0 opacity-[0.03]" style={{
      backgroundImage: 'radial-gradient(circle at 1px 1px, currentColor 1px, transparent 0)',
      backgroundSize: '28px 28px',
    }} />

    <div className="relative max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
      <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-orange-50 border border-orange-100 mb-6 opacity-0 animate-[fadeInUp_0.6s_ease-out_0.2s_forwards]">
        <Layers className="w-3.5 h-3.5 text-orange" />
        <span className="text-xs font-semibold text-orange-dark tracking-wide uppercase">Product Overview</span>
      </div>

      <h1 className="text-4xl lg:text-5xl font-bold text-content-primary mb-6 leading-tight opacity-0 animate-[fadeInUp_0.7s_ease-out_0.4s_forwards]">
        One platform.
        <br />
        <span className="text-transparent bg-clip-text bg-gradient-to-r from-orange via-orange-dark to-orange animate-shimmer bg-[length:200%_auto]">
          Complete logistics intelligence.
        </span>
      </h1>

      <p className="text-lg text-content-secondary leading-relaxed max-w-2xl mx-auto mb-10 opacity-0 animate-[fadeInUp_0.7s_ease-out_0.6s_forwards]">
        From raw CSV uploads to actionable bottleneck insights — Vyn's AI-powered pipeline handles everything in under 5 minutes. No data science team required.
      </p>

      <div className="flex flex-col sm:flex-row items-center justify-center gap-3 opacity-0 animate-[fadeInUp_0.7s_ease-out_0.8s_forwards]">
        <Link
          to="/demo"
          className="flex items-center gap-2 px-6 py-3.5 bg-orange hover:bg-orange-dark rounded-xl text-base font-semibold text-white transition-all shadow-lg hover:shadow-xl hover:shadow-orange/20 active:scale-[0.97]"
        >
          See It In Action
          <ArrowRight className="w-4 h-4" />
        </Link>
        <Link
          to="/product/how-it-works"
          className="flex items-center gap-2 px-6 py-3.5 border border-border hover:border-navy/30 rounded-xl text-base font-medium text-navy hover:bg-navy-50 transition-all"
        >
          How It Works
          <ArrowRight className="w-4 h-4" />
        </Link>
      </div>
    </div>
  </section>
);

/* ──────────────────────────────────────────
   Section 2: Pipeline Overview
   ────────────────────────────────────────── */

const pipelineSteps = [
  {
    number: '01',
    icon: <Upload className="w-5 h-5" />,
    title: 'Upload',
    subtitle: 'CSV event data',
    color: 'bg-navy',
  },
  {
    number: '02',
    icon: <Columns3 className="w-5 h-5" />,
    title: 'AI Mapping',
    subtitle: 'Auto-detect columns',
    color: 'bg-cyan',
  },
  {
    number: '03',
    icon: <GitBranch className="w-5 h-5" />,
    title: 'Flow Detection',
    subtitle: 'Classify workflow',
    color: 'bg-orange',
  },
  {
    number: '04',
    icon: <Settings className="w-5 h-5" />,
    title: 'Engineering',
    subtitle: 'Extract features',
    color: 'bg-navy',
  },
  {
    number: '05',
    icon: <Cpu className="w-5 h-5" />,
    title: 'ML Detection',
    subtitle: 'Isolation Forest',
    color: 'bg-cyan',
  },
  {
    number: '06',
    icon: <Target className="w-5 h-5" />,
    title: 'Bottlenecks',
    subtitle: 'Pinpoint delays',
    color: 'bg-orange',
  },
  {
    number: '07',
    icon: <BarChart3 className="w-5 h-5" />,
    title: 'Dashboard',
    subtitle: 'Insights & reports',
    color: 'bg-navy',
  },
];

const PipelineSection = () => (
  <section className="py-16 lg:py-24 bg-white border-b border-border">
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
      {/* Header */}
      <div className="text-center max-w-2xl mx-auto mb-12 lg:mb-16">
        <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-navy-50 border border-navy-100 mb-4">
          <Settings className="w-3.5 h-3.5 text-navy" />
          <span className="text-xs font-semibold text-navy tracking-wide uppercase">The Pipeline</span>
        </div>
        <h2 className="text-3xl lg:text-4xl font-bold text-content-primary mb-4">
          7 steps from raw data to insights
        </h2>
        <p className="text-lg text-content-secondary leading-relaxed">
          Every step is automated. You upload, confirm, and let AI do the heavy lifting.
        </p>
      </div>

      {/* Pipeline Flow */}
      <div className="relative">
        {/* Connector line (desktop) */}
        <div className="hidden lg:block absolute top-12 left-[8%] right-[8%] h-[2px] bg-gradient-to-r from-navy/20 via-orange/30 to-navy/20" />

        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-7 gap-4 lg:gap-3">
          {pipelineSteps.map((step, i) => (
            <div
              key={step.number}
              className="relative group flex flex-col items-center text-center"
              style={{ animationDelay: `${i * 100}ms` }}
            >
              {/* Step circle */}
              <div className={`relative z-10 w-14 h-14 rounded-2xl ${step.color} text-white flex items-center justify-center mb-3 shadow-md group-hover:scale-110 transition-transform duration-300`}>
                {step.icon}
              </div>

              {/* Step number */}
              <span className="text-[10px] font-bold text-content-muted uppercase tracking-widest mb-1">
                Step {step.number}
              </span>

              {/* Title */}
              <h3 className="text-sm font-semibold text-content-primary mb-0.5">
                {step.title}
              </h3>
              <p className="text-xs text-content-muted leading-snug">
                {step.subtitle}
              </p>
            </div>
          ))}
        </div>
      </div>

      {/* Link to full walkthrough */}
      <div className="text-center mt-10">
        <Link
          to="/product/how-it-works"
          className="inline-flex items-center gap-2 text-sm font-semibold text-orange hover:text-orange-dark transition-colors"
        >
          See the full walkthrough
          <ArrowRight className="w-4 h-4" />
        </Link>
      </div>
    </div>
  </section>
);

/* ──────────────────────────────────────────
   Section 3: Product Areas (cards → sub-pages)
   ────────────────────────────────────────── */

const productAreas = [
  {
    icon: <Settings className="w-6 h-6" />,
    title: 'How It Works',
    description: 'Walk through the complete 7-step pipeline: from CSV upload and AI column mapping to automated anomaly detection and actionable insights.',
    href: '/product/how-it-works',
    color: 'text-navy',
    bg: 'bg-navy-50',
    tag: 'Pipeline',
  },
  {
    icon: <Cpu className="w-6 h-6" />,
    title: 'AI Process Intelligence',
    description: 'Isolation Forest anomaly detection, smart column mapping, and automated process flow classification — all powered by unsupervised ML.',
    href: '/product/ai-intelligence',
    color: 'text-cyan',
    bg: 'bg-cyan-50',
    tag: 'ML Engine',
  },
  {
    icon: <Target className="w-6 h-6" />,
    title: 'Bottleneck Detection',
    description: 'Pinpoint the exact process steps causing delays. Each bottleneck gets a risk score, deviation analysis, and severity rating.',
    href: '/product/bottleneck-detection',
    color: 'text-orange',
    bg: 'bg-orange-50',
    tag: 'Core Feature',
  },
  {
    icon: <BarChart3 className="w-6 h-6" />,
    title: 'Process Analytics',
    description: 'Real-time dashboards with process flow visualization, duration heatmaps, throughput charts, and risk trend monitoring.',
    href: '/product/analytics',
    color: 'text-success',
    bg: 'bg-success-50',
    tag: 'Dashboard',
  },
  {
    icon: <Zap className="w-6 h-6" />,
    title: 'Technology',
    description: 'Built with React, TypeScript, Python, and Scikit-learn. Explore our architecture, tech stack, and open-source foundations.',
    href: '/product/technology',
    color: 'text-anomaly',
    bg: 'bg-anomaly-50',
    tag: 'Stack',
  },
];

const ProductAreasSection = () => (
  <section className="py-16 lg:py-24 bg-surface">
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
      {/* Header */}
      <div className="text-center max-w-2xl mx-auto mb-12 lg:mb-16">
        <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-cyan-50 border border-cyan-100 mb-4">
          <Sparkles className="w-3.5 h-3.5 text-cyan" />
          <span className="text-xs font-semibold text-cyan-dark tracking-wide uppercase">Explore</span>
        </div>
        <h2 className="text-3xl lg:text-4xl font-bold text-content-primary mb-4">
          Dive deeper into each area
        </h2>
        <p className="text-lg text-content-secondary leading-relaxed">
          Each capability is designed to work together — or explore them individually.
        </p>
      </div>

      {/* Cards grid — 2 cols on md, first row 3 cols on lg, second row 2 centered */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {productAreas.map((area) => (
          <Link
            key={area.href}
            to={area.href}
            className="group relative bg-white rounded-2xl border border-border p-6 hover:shadow-elevated hover:border-orange/20 transition-all duration-300 flex flex-col"
          >
            {/* Tag */}
            <span className={`inline-flex self-start px-2 py-0.5 rounded-md text-[10px] font-semibold uppercase tracking-wider ${area.bg} ${area.color} mb-4`}>
              {area.tag}
            </span>

            {/* Icon */}
            <div className={`w-12 h-12 rounded-xl ${area.bg} ${area.color} flex items-center justify-center mb-4 group-hover:scale-110 transition-transform duration-300`}>
              {area.icon}
            </div>

            {/* Content */}
            <h3 className="text-lg font-semibold text-content-primary mb-2 flex items-center gap-2">
              {area.title}
              <ArrowRight className="w-4 h-4 text-content-muted opacity-0 group-hover:opacity-100 group-hover:translate-x-1 transition-all duration-300" />
            </h3>
            <p className="text-sm text-content-secondary leading-relaxed flex-1">
              {area.description}
            </p>
          </Link>
        ))}
      </div>
    </div>
  </section>
);

/* ──────────────────────────────────────────
   Section 4: Key Numbers
   ────────────────────────────────────────── */

const stats = [
  { value: '< 5 min', label: 'Time to first insight', icon: <Clock className="w-5 h-5" /> },
  { value: '94%', label: 'Detection accuracy', icon: <Target className="w-5 h-5" /> },
  { value: '3 types', label: 'Supported workflows', icon: <GitBranch className="w-5 h-5" /> },
  { value: '40%', label: 'Average delay reduction', icon: <TrendingUp className="w-5 h-5" /> },
];

const StatsSection = () => (
  <section className="py-16 lg:py-20 bg-navy text-white relative overflow-hidden">
    <div className="absolute inset-0 opacity-5">
      <div className="absolute inset-0" style={{
        backgroundImage: 'radial-gradient(circle at 1px 1px, white 1px, transparent 0)',
        backgroundSize: '40px 40px',
      }} />
    </div>
    <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-8 lg:gap-12">
        {stats.map((stat) => (
          <div key={stat.label} className="text-center">
            <div className="w-10 h-10 rounded-lg bg-white/10 flex items-center justify-center mx-auto mb-3 text-orange">
              {stat.icon}
            </div>
            <p className="text-3xl lg:text-4xl font-extrabold text-white mb-1">
              {stat.value}
            </p>
            <p className="text-sm text-slate-400">
              {stat.label}
            </p>
          </div>
        ))}
      </div>
    </div>
  </section>
);

/* ──────────────────────────────────────────
   Section 5: Supported Workflows
   ────────────────────────────────────────── */

const workflows = [
  {
    icon: '🚛',
    title: 'Trucking Delivery',
    keywords: ['TRANSIT', 'DELIVERY', 'DISPATCH'],
    description: 'Optimize last-mile routes, detect loading delays, and reduce delivery failures.',
    steps: ['Dispatch', 'Loading', 'Transit', 'Delivery', 'Confirmation'],
  },
  {
    icon: '🏭',
    title: 'Warehouse Fulfillment',
    keywords: ['PACKING', 'STORAGE', 'PICKING'],
    description: 'Cut pick-pack-ship cycle times and identify staging bottlenecks.',
    steps: ['Receiving', 'Storage', 'Picking', 'Packing', 'Shipping'],
  },
  {
    icon: '🚢',
    title: 'Import Customs Clearance',
    keywords: ['CUSTOMS', 'INSPECTION', 'PORT'],
    description: 'Streamline port handling, customs inspection, and documentation workflows.',
    steps: ['Port Arrival', 'Documentation', 'Customs Inspection', 'Clearance', 'Release'],
  },
];

const WorkflowsSection = () => (
  <section className="py-16 lg:py-24 bg-white">
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
      {/* Header */}
      <div className="text-center max-w-2xl mx-auto mb-12 lg:mb-16">
        <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-orange-50 border border-orange-100 mb-4">
          <GitBranch className="w-3.5 h-3.5 text-orange" />
          <span className="text-xs font-semibold text-orange-dark tracking-wide uppercase">Auto-Detected Workflows</span>
        </div>
        <h2 className="text-3xl lg:text-4xl font-bold text-content-primary mb-4">
          3 logistics workflows, auto-classified
        </h2>
        <p className="text-lg text-content-secondary leading-relaxed">
          Vyn analyzes your step names and sequences to automatically detect which workflow type your data represents.
        </p>
      </div>

      {/* Workflow cards */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {workflows.map((flow) => (
          <div
            key={flow.title}
            className="group bg-white rounded-2xl border border-border p-6 hover:shadow-elevated hover:border-orange/20 transition-all duration-300"
          >
            {/* Icon + title */}
            <div className="flex items-center gap-3 mb-4">
              <span className="text-3xl">{flow.icon}</span>
              <h3 className="text-lg font-semibold text-content-primary">
                {flow.title}
              </h3>
            </div>

            {/* Description */}
            <p className="text-sm text-content-secondary leading-relaxed mb-5">
              {flow.description}
            </p>

            {/* Keywords detected */}
            <div className="mb-5">
              <p className="text-[10px] font-semibold text-content-muted uppercase tracking-wider mb-2">
                Detection keywords
              </p>
              <div className="flex flex-wrap gap-1.5">
                {flow.keywords.map((kw) => (
                  <span
                    key={kw}
                    className="px-2 py-0.5 bg-orange-50 text-orange-dark text-[11px] font-mono font-medium rounded-md border border-orange-100"
                  >
                    {kw}
                  </span>
                ))}
              </div>
            </div>

            {/* Process steps */}
            <div>
              <p className="text-[10px] font-semibold text-content-muted uppercase tracking-wider mb-2">
                Typical process steps
              </p>
              <div className="flex flex-wrap items-center gap-1">
                {flow.steps.map((step, i) => (
                  <span key={step} className="flex items-center gap-1">
                    <span className="px-2 py-1 bg-navy-50 text-navy text-[11px] font-medium rounded-md">
                      {step}
                    </span>
                    {i < flow.steps.length - 1 && (
                      <ArrowRight className="w-3 h-3 text-content-muted" />
                    )}
                  </span>
                ))}
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  </section>
);

/* ──────────────────────────────────────────
   Section 6: CTA
   ────────────────────────────────────────── */

const CTASection = () => (
  <section className="py-16 lg:py-24 bg-surface">
    <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
      <div className="bg-gradient-to-br from-navy to-navy-dark rounded-3xl p-10 lg:p-16 relative overflow-hidden">
        {/* Background decoration */}
        <div className="absolute top-0 right-0 w-64 h-64 bg-orange/10 rounded-full blur-3xl" />
        <div className="absolute bottom-0 left-0 w-48 h-48 bg-cyan/10 rounded-full blur-3xl" />

        <div className="relative">
          <h2 className="text-3xl lg:text-4xl font-bold text-white mb-4">
            Ready to see your data in action?
          </h2>
          <p className="text-lg text-slate-300 mb-8 max-w-xl mx-auto leading-relaxed">
            Upload your first CSV and get bottleneck insights in under 5 minutes. No setup, no credit card.
          </p>

          <div className="flex flex-col sm:flex-row items-center justify-center gap-3">
            <Link
              to="/register"
              className="flex items-center gap-2 px-8 py-4 bg-orange hover:bg-orange-dark rounded-xl text-base font-semibold text-white transition-all shadow-lg hover:shadow-xl hover:shadow-orange/30 active:scale-[0.97] w-full sm:w-auto justify-center"
            >
              Get Started Free
              <ArrowRight className="w-4 h-4" />
            </Link>
            <Link
              to="/demo"
              className="flex items-center gap-2 px-8 py-4 bg-white/10 hover:bg-white/15 border border-white/20 rounded-xl text-base font-medium text-white transition-all w-full sm:w-auto justify-center"
            >
              Try Interactive Demo
            </Link>
          </div>

          <div className="flex flex-wrap items-center justify-center gap-x-6 gap-y-2 mt-8 text-sm text-slate-400">
            <span className="flex items-center gap-1.5">
              <CheckCircle2 className="w-4 h-4 text-success" />
              Free 14-day trial
            </span>
            <span className="flex items-center gap-1.5">
              <Shield className="w-4 h-4 text-success" />
              No credit card required
            </span>
            <span className="flex items-center gap-1.5">
              <Clock className="w-4 h-4 text-success" />
              Setup in under 5 minutes
            </span>
          </div>
        </div>
      </div>
    </div>
  </section>
);

/* ──────────────────────────────────────────
   Product Page (All Sections Combined)
   ────────────────────────────────────────── */

const ProductPage = () => {
  return (
    <>
      <HeroSection />
      <PipelineSection />
      <ProductAreasSection />
      <StatsSection />
      <WorkflowsSection />
      <CTASection />
    </>
  );
};

export default ProductPage;
