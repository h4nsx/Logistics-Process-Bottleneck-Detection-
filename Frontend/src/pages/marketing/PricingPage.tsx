import { useState } from 'react';
import { Link } from 'react-router-dom';
import {
  Check,
  Minus,
  ChevronDown,
  ArrowRight,
  Zap,
  Star,
  Building2,
  Upload,
  BarChart3,
  Shield,
  FileText,
  HelpCircle,
  ChevronRight,
} from 'lucide-react';

/* ──────────────────────────────────────────
   Types
   ────────────────────────────────────────── */

type BillingCycle = 'monthly' | 'yearly';

interface PlanFeature {
  text: string;
  included: boolean;
  note?: string;
}

interface Plan {
  id: string;
  name: string;
  tagline: string;
  icon: React.ReactNode;
  monthlyPrice: number | null; // null = custom
  yearlyPrice: number | null;
  badge?: string;
  highlight: boolean; // orange border + glow
  cta: string;
  ctaHref: string;
  accentColor: string;
  features: PlanFeature[];
}

/* ──────────────────────────────────────────
   Pricing Data — MVP-accurate
   ────────────────────────────────────────── */

const plans: Plan[] = [
  {
    id: 'starter',
    name: 'Starter',
    tagline: 'Test the waters',
    icon: <Zap className="w-5 h-5" />,
    monthlyPrice: 0,
    yearlyPrice: 0,
    highlight: false,
    cta: 'Get Started Free',
    ctaHref: '/register',
    accentColor: 'text-cyan',
    features: [
      { text: '1 User seat', included: true },
      { text: 'Up to 5 CSV uploads / month', included: true },
      { text: 'Basic bottleneck detection (Mean & p95)', included: true },
      { text: 'Risk % score per step', included: true },
      { text: '5-column CSV schema support', included: true },
      { text: 'Data validation & error reporting', included: true },
      { text: 'Full Dashboard & Timeline views', included: false },
      { text: 'Advanced Z-Score anomaly rules', included: false },
      { text: 'Exportable reports (PDF / CSV)', included: false },
      { text: 'Custom risk thresholds', included: false },
      { text: 'Dedicated support', included: false },
    ],
  },
  {
    id: 'professional',
    name: 'Professional',
    tagline: 'For active operations teams',
    icon: <Star className="w-5 h-5" />,
    monthlyPrice: 49,
    yearlyPrice: 39,
    badge: 'Most Popular',
    highlight: true,
    cta: 'Start Free Trial',
    ctaHref: '/register',
    accentColor: 'text-orange',
    features: [
      { text: 'Up to 5 User seats', included: true },
      { text: 'Up to 50 CSV uploads / month', included: true },
      { text: 'Basic bottleneck detection (Mean & p95)', included: true },
      { text: 'Risk % score per step', included: true },
      { text: '5-column CSV schema support', included: true },
      { text: 'Data validation & error reporting', included: true },
      { text: 'Full Dashboard & Timeline views', included: true },
      { text: 'Advanced Z-Score anomaly rules', included: true },
      { text: 'Exportable reports (PDF / CSV)', included: true },
      { text: 'Custom risk thresholds', included: false },
      { text: 'Dedicated support', included: false },
    ],
  },
  {
    id: 'enterprise',
    name: 'Enterprise',
    tagline: 'For large supply chains',
    icon: <Building2 className="w-5 h-5" />,
    monthlyPrice: null,
    yearlyPrice: null,
    highlight: false,
    cta: 'Contact Sales',
    ctaHref: '/about-us/contact',
    accentColor: 'text-navy',
    features: [
      { text: 'Unlimited User seats', included: true },
      { text: 'Unlimited CSV uploads', included: true },
      { text: 'Basic bottleneck detection (Mean & p95)', included: true },
      { text: 'Risk % score per step', included: true },
      { text: '5-column CSV schema support', included: true },
      { text: 'Data validation & error reporting', included: true },
      { text: 'Full Dashboard & Timeline views', included: true },
      { text: 'Advanced Z-Score anomaly rules', included: true },
      { text: 'Exportable reports (PDF / CSV)', included: true },
      { text: 'Custom risk thresholds', included: true },
      { text: 'Dedicated support & SLA', included: true },
    ],
  },
];

/* ──────────────────────────────────────────
   FAQ Data — MVP-accurate
   ────────────────────────────────────────── */

interface FAQ {
  q: string;
  a: string;
}

const pricingFaqs: FAQ[] = [
  {
    q: 'Do I need any technical skills to use Vyn?',
    a: 'No technical skills are required. If your operations team can export data to a CSV file, you can use Vyn. Simply upload your execution timestamps in the required 5-column format and the platform handles all the statistical analysis automatically.',
  },
  {
    q: 'What exactly is a "CSV upload"?',
    a: 'A CSV upload is when you export your process execution records from your TMS, WMS, or ERP as a standard comma-separated values file. Vyn reads this file, validates it, and computes bottleneck baselines. No API connections or integrations are required.',
  },
  {
    q: 'Does Vyn make decisions or optimise my supply chain automatically?',
    a: 'No. Vyn is a decision-support tool, not an automation engine. It surfaces statistical evidence of where delays are occurring so your operations team can investigate and act. It never modifies routes, schedules, or external systems.',
  },
  {
    q: 'Does the AI predict future delays?',
    a: 'No. Vyn uses historical execution data only. It detects anomalies that have already occurred by comparing step durations to their historical baselines (mean, standard deviation, and 95th percentile). It does not forecast future events.',
  },
  {
    q: 'Can I switch plans later?',
    a: 'Yes. You can upgrade, downgrade, or cancel your plan any time from your account settings. Upgrades take effect immediately; downgrades apply at the start of your next billing cycle.',
  },
  {
    q: 'What data do I need to share with Vyn?',
    a: 'Only execution timestamps and step identifiers: process_id, step_code, start_time, end_time, and location. Vyn never asks for pricing data, contract values, customer names, or any sensitive business information.',
  },
];

/* ──────────────────────────────────────────
   FAQ Accordion Item
   ────────────────────────────────────────── */

const FAQItem = ({ faq, isOpen, onToggle }: { faq: FAQ; isOpen: boolean; onToggle: () => void }) => (
  <div className={`rounded-xl border transition-all duration-200 ${isOpen ? 'border-orange/20 shadow-card bg-white' : 'border-border bg-white hover:border-border-dark'}`}>
    <button
      onClick={onToggle}
      className="w-full flex items-center justify-between gap-4 px-5 py-4 text-left"
    >
      <span className={`text-sm font-semibold leading-snug transition-colors ${isOpen ? 'text-navy' : 'text-content-primary'}`}>
        {faq.q}
      </span>
      <ChevronDown className={`w-4 h-4 text-content-muted shrink-0 transition-transform duration-300 ${isOpen ? 'rotate-180 text-orange' : ''}`} />
    </button>
    {isOpen && (
      <div className="px-5 pb-5 border-t border-border/60 pt-3 animate-fade-in">
        <p className="text-sm text-content-secondary leading-relaxed">{faq.a}</p>
      </div>
    )}
  </div>
);

/* ──────────────────────────────────────────
   Pricing Card
   ────────────────────────────────────────── */

const PricingCard = ({ plan, billing }: { plan: Plan; billing: BillingCycle }) => {
  const price = billing === 'monthly' ? plan.monthlyPrice : plan.yearlyPrice;
  const isCustom = price === null;

  return (
    <div
      className={`relative flex flex-col rounded-2xl border bg-white transition-all duration-300 ${
        plan.highlight
          ? 'border-orange shadow-[0_0_0_2px_#F97316,0_8px_32px_-4px_rgba(249,115,22,0.25)]'
          : 'border-border hover:border-border-dark hover:shadow-elevated'
      }`}
    >
      {/* Most Popular badge */}
      {plan.badge && (
        <div className="absolute -top-3.5 left-1/2 -translate-x-1/2">
          <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-bold bg-orange text-white shadow-sm">
            <Star className="w-3 h-3" />
            {plan.badge}
          </span>
        </div>
      )}

      {/* Plan header */}
      <div className={`p-6 pb-4 ${plan.highlight ? 'pt-8' : ''}`}>
        <div className={`w-10 h-10 rounded-xl flex items-center justify-center mb-3 ${
          plan.highlight ? 'bg-orange/15 text-orange' : plan.id === 'enterprise' ? 'bg-navy/10 text-navy' : 'bg-cyan/10 text-cyan'
        }`}>
          {plan.icon}
        </div>

        <h3 className="text-lg font-bold text-navy mb-0.5">{plan.name}</h3>
        <p className="text-sm text-content-muted mb-4">{plan.tagline}</p>

        {/* Price */}
        <div className="mb-5">
          {isCustom ? (
            <div>
              <span className="text-4xl font-black text-navy">Custom</span>
              <p className="text-xs text-content-muted mt-1">Tailored to your team size</p>
            </div>
          ) : (
            <div className="flex items-end gap-1">
              <span className="text-4xl font-black text-navy">
                {price === 0 ? 'Free' : `$${price}`}
              </span>
              {price! > 0 && (
                <span className="text-content-muted text-sm mb-1.5">
                  / user / mo
                  {billing === 'yearly' && <span className="ml-1 text-xs text-success font-semibold">billed yearly</span>}
                </span>
              )}
            </div>
          )}
        </div>

        {/* CTA */}
        <Link
          to={plan.ctaHref}
          className={`flex items-center justify-center gap-2 w-full px-4 py-2.5 rounded-xl text-sm font-semibold transition-all duration-150 ${
            plan.highlight
              ? 'bg-orange hover:bg-orange-dark text-white shadow-sm hover:shadow-md hover:shadow-orange/20 active:scale-[0.97]'
              : plan.id === 'enterprise'
              ? 'bg-navy hover:bg-navy-light text-white active:scale-[0.97]'
              : 'bg-surface border border-border hover:border-border-dark text-navy hover:bg-surface-dark'
          }`}
        >
          {plan.cta}
          <ArrowRight className="w-4 h-4" />
        </Link>
      </div>

      {/* Divider */}
      <div className="mx-6 border-t border-border" />

      {/* Feature list */}
      <div className="p-6 pt-5 flex-1">
        <p className="text-[10px] font-bold uppercase tracking-widest text-content-muted mb-3">
          What's included
        </p>
        <ul className="space-y-2.5">
          {plan.features.map((feat) => (
            <li key={feat.text} className="flex items-start gap-2.5">
              {feat.included ? (
                <Check className="w-4 h-4 text-success shrink-0 mt-0.5" />
              ) : (
                <Minus className="w-4 h-4 text-content-muted shrink-0 mt-0.5" />
              )}
              <span className={`text-sm leading-snug ${feat.included ? 'text-content-secondary' : 'text-content-muted'}`}>
                {feat.text}
                {feat.note && <span className="ml-1 text-xs text-content-muted">({feat.note})</span>}
              </span>
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
};

/* ──────────────────────────────────────────
   Main Pricing Page
   ────────────────────────────────────────── */

const PricingPage = () => {
  const [billing, setBilling] = useState<BillingCycle>('monthly');
  const [openFaqId, setOpenFaqId] = useState<number | null>(null);

  const yearlySavings = 20; // %

  return (
    <div className="min-h-screen bg-surface pt-20">

      {/* ── Hero Header ── */}
      <div className="bg-white border-b border-border">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-12 text-center">
          {/* Breadcrumb */}
          <div className="flex items-center justify-center gap-2 text-sm text-content-muted mb-5">
            <HelpCircle className="w-4 h-4" />
            <span>Pricing</span>
          </div>

          <h1 className="text-4xl sm:text-5xl font-bold text-navy mb-4 leading-tight">
            Simple, transparent pricing
          </h1>
          <p className="text-lg text-content-secondary max-w-xl mx-auto mb-8 leading-relaxed">
            Start for free with CSV-based bottleneck detection. Scale up when
            your operations team needs advanced analytics and more uploads.
          </p>

          {/* Billing toggle */}
          <div className="inline-flex items-center gap-3 bg-surface border border-border rounded-xl p-1 shadow-card">
            <button
              onClick={() => setBilling('monthly')}
              className={`px-4 py-2 rounded-lg text-sm font-semibold transition-all duration-150 ${
                billing === 'monthly'
                  ? 'bg-white text-navy shadow-card'
                  : 'text-content-muted hover:text-content-secondary'
              }`}
            >
              Monthly
            </button>
            <button
              onClick={() => setBilling('yearly')}
              className={`relative px-4 py-2 rounded-lg text-sm font-semibold transition-all duration-150 ${
                billing === 'yearly'
                  ? 'bg-white text-navy shadow-card'
                  : 'text-content-muted hover:text-content-secondary'
              }`}
            >
              Yearly
              <span className="ml-1.5 inline-flex items-center px-1.5 py-0.5 rounded-full text-[10px] font-bold bg-success/15 text-success">
                -{yearlySavings}%
              </span>
            </button>
          </div>
        </div>
      </div>

      {/* ── Pricing Cards ── */}
      <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 lg:gap-8">
          {plans.map((plan) => (
            <PricingCard key={plan.id} plan={plan} billing={billing} />
          ))}
        </div>

        {/* Trust bar */}
        <div className="flex flex-wrap items-center justify-center gap-6 mt-10 text-sm text-content-muted">
          {[
            { icon: <Shield className="w-4 h-4 text-success" />, label: 'No credit card required for Starter' },
            { icon: <Upload className="w-4 h-4 text-cyan" />, label: 'CSV upload in minutes' },
            { icon: <FileText className="w-4 h-4 text-navy" />, label: 'Cancel any time' },
            { icon: <BarChart3 className="w-4 h-4 text-orange" />, label: 'Real statistical baselines, not guesswork' },
          ].map(({ icon, label }) => (
            <span key={label} className="flex items-center gap-1.5">
              {icon}
              {label}
            </span>
          ))}
        </div>
      </div>

      {/* ── Feature comparison table ── */}
      <div className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 pb-16">
        <div className="bg-white rounded-2xl border border-border shadow-card overflow-hidden">
          <div className="px-6 py-5 border-b border-border">
            <h2 className="text-lg font-bold text-navy">Full Feature Comparison</h2>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-sm min-w-[600px]">
              <thead>
                <tr className="border-b border-border">
                  <th className="text-left px-6 py-3 text-xs font-semibold uppercase tracking-wider text-content-muted w-1/2">Feature</th>
                  {plans.map((p) => (
                    <th key={p.id} className={`text-center px-4 py-3 text-xs font-bold uppercase tracking-wider ${
                      p.highlight ? 'text-orange' : p.id === 'enterprise' ? 'text-navy' : 'text-cyan'
                    }`}>
                      {p.name}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody className="divide-y divide-border">
                {[
                  { label: 'User seats', values: ['1', 'Up to 5', 'Unlimited'] },
                  { label: 'CSV uploads / month', values: ['5', '50', 'Unlimited'] },
                  { label: 'Mean & p95 baseline detection', values: [true, true, true] },
                  { label: 'Z-score anomaly rules', values: [false, true, true] },
                  { label: 'Risk % score per step', values: [true, true, true] },
                  { label: 'Full Dashboard & Timeline views', values: [false, true, true] },
                  { label: 'Exportable reports', values: [false, true, true] },
                  { label: 'Custom risk thresholds', values: [false, false, true] },
                  { label: 'Dedicated support & SLA', values: [false, false, true] },
                ].map(({ label, values }) => (
                  <tr key={label} className="hover:bg-surface/50 transition-colors">
                    <td className="px-6 py-3 text-content-secondary">{label}</td>
                    {values.map((val, i) => (
                      <td key={i} className="px-4 py-3 text-center">
                        {typeof val === 'boolean' ? (
                          val
                            ? <Check className="w-4 h-4 text-success mx-auto" />
                            : <Minus className="w-4 h-4 text-content-muted mx-auto" />
                        ) : (
                          <span className="text-xs font-semibold text-navy">{val}</span>
                        )}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      {/* ── FAQ Section ── */}
      <div className="bg-white border-t border-border py-16 lg:py-20">
        <div className="max-w-3xl mx-auto px-4 sm:px-6 lg:px-8">
          {/* Header */}
          <div className="text-center mb-10">
            <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-navy/5 border border-navy/10 mb-4">
              <HelpCircle className="w-3.5 h-3.5 text-navy" />
              <span className="text-xs font-semibold text-navy tracking-wide uppercase">FAQ</span>
            </div>
            <h2 className="text-2xl sm:text-3xl font-bold text-navy mb-3">
              Frequently asked questions
            </h2>
            <p className="text-content-secondary">
              Still unsure?{' '}
              <Link to="/about-us/contact" className="text-orange hover:text-orange-dark font-medium transition-colors">
                Talk to our team
              </Link>
            </p>
          </div>

          {/* Accordion */}
          <div className="space-y-3">
            {pricingFaqs.map((faq, i) => (
              <FAQItem
                key={i}
                faq={faq}
                isOpen={openFaqId === i}
                onToggle={() => setOpenFaqId(openFaqId === i ? null : i)}
              />
            ))}
          </div>
        </div>
      </div>

      {/* ── CTA Banner ── */}
      <div className="bg-surface border-t border-border py-16">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="bg-navy rounded-2xl p-10 text-center relative overflow-hidden">
            <div className="absolute top-0 right-0 w-64 h-64 bg-orange/10 rounded-full blur-3xl" />
            <div className="absolute bottom-0 left-0 w-48 h-48 bg-cyan/10 rounded-full blur-3xl" />
            <div className="relative">
              <h2 className="text-2xl sm:text-3xl font-bold text-white mb-3">
                Ready to find your hidden bottlenecks?
              </h2>
              <p className="text-navy-100 text-base mb-8 max-w-lg mx-auto">
                Start for free. Upload your first CSV and get a full statistical
                bottleneck report in minutes — no credit card, no code.
              </p>
              <div className="flex flex-col sm:flex-row items-center justify-center gap-3">
                <Link
                  to="/register"
                  className="inline-flex items-center gap-2 px-7 py-3 rounded-xl bg-orange hover:bg-orange-dark text-white font-bold text-sm transition-all shadow-sm hover:shadow-md active:scale-[0.97]"
                >
                  Start Free — No Credit Card
                  <ArrowRight className="w-4 h-4" />
                </Link>
                <Link
                  to="/resources/docs"
                  className="inline-flex items-center gap-2 px-7 py-3 rounded-xl bg-white/10 hover:bg-white/20 text-white font-semibold text-sm transition-all border border-white/20"
                >
                  Read the Docs
                  <ChevronRight className="w-4 h-4" />
                </Link>
              </div>
            </div>
          </div>
        </div>
      </div>

    </div>
  );
};

export default PricingPage;
