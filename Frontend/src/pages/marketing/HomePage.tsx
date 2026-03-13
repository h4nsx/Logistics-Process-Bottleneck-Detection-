import { useEffect, useRef } from 'react';
import { Link } from 'react-router-dom';
import {
  ArrowRight,
  Zap,
  Target,
  BarChart3,
  Cpu,
  Clock,
  Shield,
  TrendingUp,
  Upload,
  GitBranch,
  LineChart,
  Truck,
  Warehouse,
  Ship,
  Building2,
  CheckCircle2,
  ChevronRight,
  Brain,
  Workflow,
  Wand2,
  Bell,
  Columns3,
  Plug,
  XCircle,
  ArrowDownUp,
  Timer,
  FileSpreadsheet,
  AlertTriangle,
  Sparkles,
} from 'lucide-react';

/* ──────────────────────────────────────────
   Scroll Reveal Hook
   ────────────────────────────────────────── */

const useScrollReveal = () => {
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;

    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            // Add 'revealed' to the section itself
            entry.target.classList.add('revealed');
            // Also reveal all children with reveal classes
            entry.target.querySelectorAll('.reveal, .reveal-scale, .reveal-left, .reveal-right, .animate-count-pop').forEach((child, i) => {
              setTimeout(() => child.classList.add('revealed'), i * 80);
            });
            observer.unobserve(entry.target);
          }
        });
      },
      { threshold: 0.1, rootMargin: '0px 0px -60px 0px' }
    );

    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  return ref;
};

/* ──────────────────────────────────────────
   Section 1: Hero
   ────────────────────────────────────────── */

const HeroSection = () => (
  <section className="relative overflow-hidden">
    {/* Background gradient */}
    <div className="absolute inset-0 bg-gradient-to-br from-navy-50 via-white to-cyan-50" />
    <div className="absolute top-0 right-0 w-[600px] h-[600px] bg-orange/5 rounded-full blur-3xl -translate-y-1/2 translate-x-1/4" />
    <div className="absolute bottom-0 left-0 w-[400px] h-[400px] bg-cyan/5 rounded-full blur-3xl translate-y-1/2 -translate-x-1/4" />

    <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 pt-16 lg:pt-24 pb-12 lg:pb-20">
      <div className="flex flex-col lg:flex-row items-center gap-12 lg:gap-16">
        {/* Left: Copy */}
        <div className="flex-1 text-center lg:text-left max-w-2xl">
          {/* Badge */}
          <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-orange-50 border border-orange-100 mb-6">
            <Zap className="w-3.5 h-3.5 text-orange" />
            <span className="text-xs font-semibold text-orange-dark tracking-wide uppercase">
              AI-Powered Process Mining
            </span>
          </div>

          <h1 className="text-4xl sm:text-5xl lg:text-6xl font-extrabold tracking-tight leading-[1.1] mb-6">
            <span className="text-content-primary">Detect </span>
            <span className="text-orange">Bottlenecks</span>
            <br />
            <span className="text-content-primary">Before They </span>
            <span className="text-navy">Cost You</span>
          </h1>

          <p className="text-lg lg:text-xl text-content-secondary leading-relaxed mb-8 max-w-lg mx-auto lg:mx-0">
            Vyn uses Isolation Forest AI to analyze your logistics data, detect anomalies in real-time, and pinpoint process bottlenecks — so you can fix delays before they cascade.
          </p>

          {/* CTAs */}
          <div className="flex flex-col sm:flex-row items-center gap-3 mb-8">
            <Link
              to="/register"
              className="flex items-center gap-2 px-6 py-3.5 bg-orange hover:bg-orange-dark rounded-xl text-base font-semibold text-white transition-all shadow-lg hover:shadow-xl hover:shadow-orange/20 active:scale-[0.97] w-full sm:w-auto justify-center animate-pulse-glow"
            >
              Start Free Trial
              <ArrowRight className="w-4 h-4" />
            </Link>
            <Link
              to="/demo"
              className="flex items-center gap-2 px-6 py-3.5 border border-border hover:border-navy/30 rounded-xl text-base font-medium text-navy hover:bg-navy-50 transition-all w-full sm:w-auto justify-center"
            >
              Watch Demo
              <ChevronRight className="w-4 h-4" />
            </Link>
          </div>

          {/* Highlights */}
          <div className="flex flex-wrap items-center gap-x-5 gap-y-2 justify-center lg:justify-start text-sm text-content-muted">
            <span className="flex items-center gap-1.5">
              <CheckCircle2 className="w-4 h-4 text-success" />
              Free 14-day trial
            </span>
            <span className="flex items-center gap-1.5">
              <CheckCircle2 className="w-4 h-4 text-success" />
              No credit card needed
            </span>
            <span className="flex items-center gap-1.5">
              <CheckCircle2 className="w-4 h-4 text-success" />
              Results in under 5 min
            </span>
          </div>
        </div>

        {/* Right: Dashboard Image */}
        <div className="flex-1 w-full max-w-xl lg:max-w-none">
          <div className="relative">
            {/* Glow behind image */}
            <div className="absolute -inset-4 bg-gradient-to-r from-orange/10 via-cyan/10 to-navy/10 rounded-3xl blur-2xl" />
            <img
              src="/hero-dashboard.png"
              alt="Vyn analytics dashboard showing process flow visualization, risk scores, and bottleneck detection"
              className="relative w-full rounded-2xl shadow-elevated border border-border/50"
            />
            {/* Floating badge */}
            <div className="absolute -bottom-4 -left-4 bg-white rounded-xl shadow-elevated border border-border px-4 py-3 flex items-center gap-3 animate-float">
              <div className="w-10 h-10 rounded-lg bg-danger-50 flex items-center justify-center">
                <Target className="w-5 h-5 text-danger" />
              </div>
              <div>
                <p className="text-xs font-medium text-content-muted">Bottleneck Found</p>
                <p className="text-sm font-bold text-danger">Customs Clearance +4.8h</p>
              </div>
            </div>
            {/* Floating badge 2 */}
            <div className="absolute -top-3 -right-3 bg-white rounded-xl shadow-elevated border border-border px-4 py-3 flex items-center gap-3 animate-float-delayed hidden sm:flex">
              <div className="w-10 h-10 rounded-lg bg-success-50 flex items-center justify-center">
                <TrendingUp className="w-5 h-5 text-success" />
              </div>
              <div>
                <p className="text-xs font-medium text-content-muted">Throughput</p>
                <p className="text-sm font-bold text-success">+23% this week</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </section>
);

/* ──────────────────────────────────────────
   Section 2: Powered By — Tech Stack
   ────────────────────────────────────────── */

const techCapabilities = [
  {
    icon: <Brain className="w-5 h-5" />,
    title: 'Isolation Forest',
    description: 'Unsupervised anomaly detection that learns what "normal" looks like in your processes.',
  },
  {
    icon: <Workflow className="w-5 h-5" />,
    title: 'Process Mining',
    description: 'Automatically reconstructs workflows from raw event logs — no manual modeling.',
  },
  {
    icon: <Wand2 className="w-5 h-5" />,
    title: 'Feature Engineering',
    description: 'Auto-generates statistical features from your data to improve detection accuracy.',
  },
  {
    icon: <Bell className="w-5 h-5" />,
    title: 'Real-Time Alerts',
    description: 'Instant notifications when anomalies or bottlenecks are detected in your pipeline.',
  },
  {
    icon: <Columns3 className="w-5 h-5" />,
    title: 'Smart Column Mapping',
    description: 'AI suggests how to map your CSV columns to process mining fields automatically.',
  },
  {
    icon: <Plug className="w-5 h-5" />,
    title: 'API-First Architecture',
    description: 'RESTful API and webhooks to integrate Vyn into your existing logistics stack.',
  },
];

const PoweredBySection = () => {
  const ref = useScrollReveal();
  return (
    <section ref={ref} className="border-y border-border bg-surface/50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12 lg:py-16">
        <div className="text-center mb-10">
          <p className="text-sm font-semibold text-orange uppercase tracking-wider mb-2">
            Under the Hood
          </p>
          <h2 className="text-2xl lg:text-3xl font-bold text-content-primary">
            Powered by cutting-edge AI
          </h2>
        </div>
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
          {techCapabilities.map((tech, i) => (
            <div
              key={tech.title}
              className={`group bg-white rounded-xl border border-border hover:border-orange/30 p-4 text-center transition-all duration-300 hover:shadow-card reveal stagger-${i + 1}`}
            >
              <div className="w-10 h-10 rounded-lg bg-navy/5 text-navy flex items-center justify-center mx-auto mb-3 group-hover:bg-orange/10 group-hover:text-orange transition-colors duration-300">
                {tech.icon}
              </div>
              <h3 className="text-sm font-semibold text-content-primary mb-1">
                {tech.title}
              </h3>
              <p className="text-xs text-content-muted leading-relaxed">
                {tech.description}
              </p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
};

/* ──────────────────────────────────────────
   Section 3: Key Features
   ────────────────────────────────────────── */

const features = [
  {
    icon: <Target className="w-6 h-6" />,
    title: 'Bottleneck Detection',
    description: 'Isolation Forest AI automatically identifies anomalous process steps that cause delays across your entire logistics pipeline.',
    color: 'text-danger',
    bg: 'bg-danger-50',
  },
  {
    icon: <Cpu className="w-6 h-6" />,
    title: 'AI Process Mining',
    description: 'Upload raw CSV event logs and Vyn automatically reconstructs your process flows, maps columns, and detects step patterns.',
    color: 'text-cyan',
    bg: 'bg-cyan-50',
  },
  {
    icon: <BarChart3 className="w-6 h-6" />,
    title: 'Real-Time Analytics',
    description: 'Monitor throughput, step durations, and risk scores with live dashboards. Spot trends before they become problems.',
    color: 'text-orange',
    bg: 'bg-orange-50',
  },
  {
    icon: <Clock className="w-6 h-6" />,
    title: 'Duration Heatmaps',
    description: 'Visualize where time is spent across every process step. Identify slow stages with color-coded duration analysis.',
    color: 'text-navy',
    bg: 'bg-navy-50',
  },
  {
    icon: <Shield className="w-6 h-6" />,
    title: 'Risk Scoring',
    description: 'Every process instance receives an automated risk score. Focus attention on the shipments most likely to fail SLAs.',
    color: 'text-anomaly',
    bg: 'bg-anomaly-50',
  },
  {
    icon: <LineChart className="w-6 h-6" />,
    title: 'Process Visualization',
    description: 'Interactive workflow graphs powered by React Flow. See your actual process paths with bottleneck nodes highlighted.',
    color: 'text-success',
    bg: 'bg-success-50',
  },
];

const FeaturesSection = () => {
  const ref = useScrollReveal();
  return (
    <section ref={ref} className="py-16 lg:py-24 bg-white">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <div className="text-center max-w-2xl mx-auto mb-12 lg:mb-16">
          <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-navy-50 border border-navy-100 mb-4">
            <Zap className="w-3.5 h-3.5 text-navy" />
            <span className="text-xs font-semibold text-navy tracking-wide uppercase">Core Capabilities</span>
          </div>
          <h2 className="text-3xl lg:text-4xl font-bold text-content-primary mb-4">
            Everything you need to optimize logistics
          </h2>
          <p className="text-lg text-content-secondary leading-relaxed">
            From data upload to actionable recommendations — Vyn covers every step of the process mining pipeline.
          </p>
        </div>

        {/* Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {features.map((feature, i) => (
            <div
              key={feature.title}
              className={`group relative bg-white rounded-2xl border border-border hover:border-border-dark p-6 transition-all duration-300 hover:shadow-elevated reveal stagger-${i + 1}`}
            >
              <div className={`w-12 h-12 rounded-xl ${feature.bg} flex items-center justify-center mb-4 ${feature.color} group-hover:scale-110 transition-transform duration-300`}>
                {feature.icon}
              </div>
              <h3 className="text-lg font-semibold text-content-primary mb-2">
                {feature.title}
              </h3>
              <p className="text-sm text-content-secondary leading-relaxed">
                {feature.description}
              </p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
};

/* ──────────────────────────────────────────
   Section 4: How It Works
   ────────────────────────────────────────── */

const steps = [
  {
    number: '01',
    icon: <Upload className="w-6 h-6" />,
    title: 'Upload Your Data',
    description: 'Drop your CSV event log — shipment records, warehouse logs, delivery timestamps. Vyn auto-detects columns and suggests mappings.',
  },
  {
    number: '02',
    icon: <GitBranch className="w-6 h-6" />,
    title: 'Detect Process Flows',
    description: 'Our AI reconstructs the actual process flow from your data. Review and confirm the detected steps and transitions.',
  },
  {
    number: '03',
    icon: <Target className="w-6 h-6" />,
    title: 'Find Bottlenecks',
    description: 'Isolation Forest anomaly detection identifies the exact steps causing delays. Each bottleneck gets a risk score and severity rating.',
  },
  {
    number: '04',
    icon: <TrendingUp className="w-6 h-6" />,
    title: 'Optimize & Monitor',
    description: 'Get AI-powered recommendations to fix bottlenecks. Set up alerts for recurring anomalies and monitor improvements in real-time.',
  },
];

const HowItWorksSection = () => {
  const ref = useScrollReveal();
  return (
    <section ref={ref} className="py-16 lg:py-24 bg-surface">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <div className="text-center max-w-2xl mx-auto mb-12 lg:mb-16 reveal">
          <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-orange-50 border border-orange-100 mb-4">
            <Clock className="w-3.5 h-3.5 text-orange" />
            <span className="text-xs font-semibold text-orange-dark tracking-wide uppercase">Get Started in Minutes</span>
          </div>
          <h2 className="text-3xl lg:text-4xl font-bold text-content-primary mb-4">
            From raw data to insights in 4 steps
          </h2>
          <p className="text-lg text-content-secondary leading-relaxed">
            No complex setup. No data science team required. Just upload and go.
          </p>
        </div>

        {/* Steps */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 lg:gap-8">
          {steps.map((step, index) => (
            <div key={step.number} className={`relative reveal stagger-${index + 1}`}>
              {/* Connector line (desktop) */}
              {index < steps.length - 1 && (
                <div className="hidden lg:block absolute top-10 left-[calc(50%+40px)] w-[calc(100%-40px)] h-[2px] bg-gradient-to-r from-orange/40 to-orange/10" />
              )}
              <div className="relative bg-white rounded-2xl border border-border p-6 hover:shadow-elevated transition-all duration-300 text-center">
                <div className="text-4xl font-black text-orange/15 mb-3 select-none">
                  {step.number}
                </div>
                <div className="w-12 h-12 rounded-xl bg-navy text-white flex items-center justify-center mx-auto mb-4">
                  {step.icon}
                </div>
                <h3 className="text-lg font-semibold text-content-primary mb-2">
                  {step.title}
                </h3>
                <p className="text-sm text-content-secondary leading-relaxed">
                  {step.description}
                </p>
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
};

/* ──────────────────────────────────────────
   Section 5: Stats
   ────────────────────────────────────────── */

const stats = [
  { value: '< 5 min', label: 'Time to first insight', icon: <Clock className="w-5 h-5" /> },
  { value: '94%', label: 'Bottleneck detection accuracy', icon: <Target className="w-5 h-5" /> },
  { value: '3.2x', label: 'Faster issue resolution', icon: <Zap className="w-5 h-5" /> },
  { value: '40%', label: 'Average delay reduction', icon: <TrendingUp className="w-5 h-5" /> },
];

const StatsSection = () => {
  const ref = useScrollReveal();
  return (
    <section ref={ref} className="py-16 lg:py-20 bg-navy text-white relative overflow-hidden">
      {/* Background pattern */}
      <div className="absolute inset-0 opacity-5">
        <div className="absolute inset-0" style={{
          backgroundImage: 'radial-gradient(circle at 1px 1px, white 1px, transparent 0)',
          backgroundSize: '40px 40px',
        }} />
      </div>
      <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-8 lg:gap-12">
          {stats.map((stat) => (
            <div key={stat.label} className="text-center animate-count-pop">
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
};

/* ──────────────────────────────────────────
   Section 6: Solutions / Industries
   ────────────────────────────────────────── */

const solutions = [
  {
    icon: <Truck className="w-7 h-7" />,
    title: 'Trucking & Delivery',
    description: 'Optimize last-mile delivery routes, identify loading delays, and reduce failed delivery attempts.',
    href: '/solutions/trucking',
  },
  {
    icon: <Warehouse className="w-7 h-7" />,
    title: 'Warehouse Fulfillment',
    description: 'Cut pick-pack-ship cycle times by detecting bottlenecks in warehouse workflows and staging areas.',
    href: '/solutions/warehouse',
  },
  {
    icon: <Ship className="w-7 h-7" />,
    title: 'Import / Export',
    description: 'Streamline customs clearance, port handling, and cross-border logistics with real-time anomaly tracking.',
    href: '/solutions/import-export',
  },
  {
    icon: <Building2 className="w-7 h-7" />,
    title: 'Enterprise Monitoring',
    description: 'Multi-facility, multi-region operational intelligence with unified dashboards and alerting.',
    href: '/solutions/enterprise',
  },
];

const SolutionsSection = () => {
  const ref = useScrollReveal();
  return (
    <section ref={ref} className="py-16 lg:py-24 bg-white">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <div className="text-center max-w-2xl mx-auto mb-12 lg:mb-16 reveal">
          <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-cyan-50 border border-cyan-100 mb-4">
            <GitBranch className="w-3.5 h-3.5 text-cyan" />
            <span className="text-xs font-semibold text-cyan-dark tracking-wide uppercase">Industry Solutions</span>
          </div>
          <h2 className="text-3xl lg:text-4xl font-bold text-content-primary mb-4">
            Built for every link in the supply chain
          </h2>
          <p className="text-lg text-content-secondary leading-relaxed">
            Whether you manage fleets, warehouses, or international shipments — Vyn adapts to your process.
          </p>
        </div>

        {/* Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {solutions.map((solution, i) => (
            <Link
              key={solution.title}
              to={solution.href}
              className={`group relative bg-white rounded-2xl border border-border hover:border-orange/30 p-8 transition-all duration-300 hover:shadow-elevated flex items-start gap-5 reveal stagger-${i + 1}`}
            >
              <div className="w-14 h-14 rounded-2xl bg-navy text-orange flex items-center justify-center shrink-0 group-hover:scale-105 transition-transform duration-300">
                {solution.icon}
              </div>
              <div className="flex-1 min-w-0">
                <h3 className="text-lg font-semibold text-content-primary mb-2 flex items-center gap-2">
                  {solution.title}
                  <ArrowRight className="w-4 h-4 text-content-muted opacity-0 group-hover:opacity-100 group-hover:translate-x-1 transition-all duration-300" />
                </h3>
                <p className="text-sm text-content-secondary leading-relaxed">
                  {solution.description}
                </p>
              </div>
            </Link>
          ))}
        </div>
      </div>
    </section>
  );
};

/* ──────────────────────────────────────────
   Section 7: Before vs After Vyn
   ────────────────────────────────────────── */

const beforeItems = [
  { icon: <FileSpreadsheet className="w-5 h-5" />, text: 'Manual spreadsheet analysis for days' },
  { icon: <XCircle className="w-5 h-5" />, text: 'Bottlenecks discovered after SLA breaches' },
  { icon: <Timer className="w-5 h-5" />, text: 'Weeks to identify root causes of delays' },
  { icon: <AlertTriangle className="w-5 h-5" />, text: 'Reactive firefighting when issues escalate' },
  { icon: <ArrowDownUp className="w-5 h-5" />, text: 'Siloed data across teams and systems' },
];

const afterItems = [
  { icon: <Sparkles className="w-5 h-5" />, text: 'AI-driven detection in under 5 minutes' },
  { icon: <Target className="w-5 h-5" />, text: 'Bottlenecks flagged before they impact delivery' },
  { icon: <Zap className="w-5 h-5" />, text: 'Instant root cause analysis with risk scores' },
  { icon: <Bell className="w-5 h-5" />, text: 'Proactive alerts and automated monitoring' },
  { icon: <Workflow className="w-5 h-5" />, text: 'Unified process view from raw event data' },
];

const BeforeAfterSection = () => {
  const ref = useScrollReveal();
  return (
    <section ref={ref} className="py-16 lg:py-24 bg-surface">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <div className="text-center max-w-2xl mx-auto mb-12 lg:mb-16">
          <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-orange-50 border border-orange-100 mb-4">
            <Sparkles className="w-3.5 h-3.5 text-orange" />
            <span className="text-xs font-semibold text-orange-dark tracking-wide uppercase">The Vyn Difference</span>
          </div>
          <h2 className="text-3xl lg:text-4xl font-bold text-content-primary mb-4">
            From reactive to proactive — instantly
          </h2>
          <p className="text-lg text-content-secondary leading-relaxed">
            See how Vyn transforms the way logistics teams detect and resolve process bottlenecks.
          </p>
        </div>

        {/* Comparison Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 lg:gap-8 max-w-5xl mx-auto">
          {/* BEFORE */}
          <div className="bg-white rounded-2xl border border-border p-8 reveal-left">
            <div className="flex items-center gap-3 mb-6">
              <div className="w-10 h-10 rounded-xl bg-danger-50 flex items-center justify-center">
                <XCircle className="w-5 h-5 text-danger" />
              </div>
              <div>
                <h3 className="text-lg font-bold text-content-primary">Without Vyn</h3>
                <p className="text-xs text-content-muted">Traditional process analysis</p>
              </div>
            </div>
            <ul className="space-y-4">
              {beforeItems.map((item, i) => (
                <li key={i} className="flex items-start gap-3">
                  <span className="mt-0.5 text-danger/60 shrink-0">{item.icon}</span>
                  <span className="text-sm text-content-secondary leading-relaxed">{item.text}</span>
                </li>
              ))}
            </ul>
          </div>

          {/* AFTER */}
          <div className="bg-white rounded-2xl border-2 border-orange/20 p-8 relative overflow-hidden reveal-right">
            {/* Glow */}
            <div className="absolute top-0 right-0 w-32 h-32 bg-orange/5 rounded-full blur-2xl" />
            <div className="relative">
              <div className="flex items-center gap-3 mb-6">
                <div className="w-10 h-10 rounded-xl bg-orange-50 flex items-center justify-center">
                  <Sparkles className="w-5 h-5 text-orange" />
                </div>
                <div>
                  <h3 className="text-lg font-bold text-content-primary">With Vyn</h3>
                  <p className="text-xs text-orange font-medium">AI-powered intelligence</p>
                </div>
              </div>
              <ul className="space-y-4">
                {afterItems.map((item, i) => (
                  <li key={i} className="flex items-start gap-3">
                    <span className="mt-0.5 text-orange shrink-0">{item.icon}</span>
                    <span className="text-sm text-content-primary font-medium leading-relaxed">{item.text}</span>
                  </li>
                ))}
              </ul>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

/* ──────────────────────────────────────────
   Section 8: Final CTA
   ────────────────────────────────────────── */

const FinalCTASection = () => {
  const ref = useScrollReveal();
  return (
    <section ref={ref} className="py-16 lg:py-24 bg-white">
      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
        <div className="bg-gradient-to-br from-navy to-navy-dark rounded-3xl p-10 lg:p-16 relative overflow-hidden reveal-scale">
          {/* Background decoration */}
          <div className="absolute top-0 right-0 w-64 h-64 bg-orange/10 rounded-full blur-3xl" />
          <div className="absolute bottom-0 left-0 w-48 h-48 bg-cyan/10 rounded-full blur-3xl" />

          <div className="relative">
            <h2 className="text-3xl lg:text-4xl font-bold text-white mb-4">
              Stop guessing. Start detecting.
            </h2>
            <p className="text-lg text-slate-300 mb-8 max-w-xl mx-auto leading-relaxed">
              Join 500+ logistics teams using Vyn to find bottlenecks faster,
              reduce delays, and optimize every step of their supply chain.
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
                <CheckCircle2 className="w-4 h-4 text-success" />
                No credit card required
              </span>
              <span className="flex items-center gap-1.5">
                <CheckCircle2 className="w-4 h-4 text-success" />
                Setup in under 5 minutes
              </span>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

/* ──────────────────────────────────────────
   HomePage (All Sections Combined)
   ────────────────────────────────────────── */

const HomePage = () => {
  return (
    <>
      <HeroSection />
      <PoweredBySection />
      <FeaturesSection />
      <HowItWorksSection />
      <StatsSection />
      <SolutionsSection />
      <BeforeAfterSection />
      <FinalCTASection />
    </>
  );
};

export default HomePage;