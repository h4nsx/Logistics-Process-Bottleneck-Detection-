import { Outlet, Link } from 'react-router-dom';
import { Target, BarChart3, Shield, CheckCircle2 } from 'lucide-react';

const brandFeatures = [
  {
    icon: <Target className="w-5 h-5 text-orange" />,
    title: 'Bottleneck Detection',
    description: 'Pinpoint delays with Isolation Forest AI',
  },
  {
    icon: <BarChart3 className="w-5 h-5 text-cyan-light" />,
    title: 'Process Analytics',
    description: 'Real-time dashboards and heatmaps',
  },
  {
    icon: <Shield className="w-5 h-5 text-success-light" />,
    title: 'Risk Scoring',
    description: 'Automated SLA risk assessment',
  },
];

const AuthLayout = () => {
  return (
    <div className="min-h-screen flex">

      {/* ── Left Panel: Branding ─────────────── */}
      <div className="hidden lg:flex lg:w-[480px] xl:w-[540px] shrink-0 bg-gradient-to-br from-navy via-navy-dark to-navy-900 relative overflow-hidden flex-col justify-between p-10 xl:p-12">
        {/* Background decoration */}
        <div className="absolute top-0 right-0 w-72 h-72 bg-orange/8 rounded-full blur-3xl" />
        <div className="absolute bottom-0 left-0 w-56 h-56 bg-cyan/8 rounded-full blur-3xl" />
        <div className="absolute inset-0 opacity-[0.03]" style={{
          backgroundImage: 'radial-gradient(circle at 1px 1px, white 1px, transparent 0)',
          backgroundSize: '32px 32px',
        }} />

        {/* Top: Logo */}
        <div className="relative">
          <Link to="/" className="flex items-center gap-2.5">
            <img src="/logo.svg" alt="Vyn" className="w-9 h-9" />
            <div className="flex flex-col">
              <span className="text-lg font-bold text-white leading-tight tracking-tight">
                Vyn
              </span>
              <span className="text-[10px] font-medium text-slate-400 leading-none tracking-wider uppercase">
                Logistics Intelligence
              </span>
            </div>
          </Link>
        </div>

        {/* Center: Tagline + Features */}
        <div className="relative -mt-8">
          <h1 className="text-3xl xl:text-4xl font-bold text-white leading-tight mb-3">
            Detect anomalies.
            <br />
            <span className="text-orange">Optimize everything.</span>
          </h1>
          <p className="text-slate-400 text-base leading-relaxed mb-10 max-w-sm">
            AI-powered process mining that identifies bottlenecks in your logistics pipeline before they cost you.
          </p>

          <div className="space-y-5">
            {brandFeatures.map((feature) => (
              <div key={feature.title} className="flex items-start gap-3.5">
                <div className="w-10 h-10 rounded-xl bg-white/[0.07] flex items-center justify-center shrink-0 ring-1 ring-white/10">
                  {feature.icon}
                </div>
                <div>
                  <p className="text-sm font-semibold text-white mb-0.5">{feature.title}</p>
                  <p className="text-xs text-slate-400">{feature.description}</p>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Bottom: Trust line */}
        <div className="relative flex items-center gap-2 text-xs text-slate-500">
          <CheckCircle2 className="w-3.5 h-3.5 text-success" />
          <span>SOC 2 Type II compliant · 256-bit encryption · 99.9% uptime</span>
        </div>
      </div>

      {/* ── Right Panel: Form ────────────────── */}
      <div className="flex-1 flex flex-col bg-white px-4 sm:px-6">
        {/* Form area — centered vertically */}
        <div className="flex-1 flex items-center justify-center py-10 lg:py-0">
          <div className="w-full max-w-md">
            {/* Mobile logo (only shown on small screens) */}
            <div className="lg:hidden flex items-center justify-center gap-2.5 mb-8">
              <Link to="/" className="flex items-center gap-2.5">
                <img src="/logo.svg" alt="Vyn" className="w-8 h-8" />
                <span className="text-lg font-bold text-navy">Vyn</span>
              </Link>
            </div>
            <Outlet />
          </div>
        </div>

        {/* Footer */}
        <footer className="py-4 border-t border-border">
          <div className="flex flex-col sm:flex-row items-center justify-between gap-2 max-w-md mx-auto w-full text-xs text-content-muted">
            <span>© {new Date().getFullYear()} Vyn. All rights reserved.</span>
            <div className="flex items-center gap-4">
              <Link to="/terms" className="hover:text-content-secondary transition-colors">Terms</Link>
              <Link to="/privacy" className="hover:text-content-secondary transition-colors">Privacy</Link>
              <Link to="/resources/documentation" className="hover:text-content-secondary transition-colors">Help</Link>
            </div>
          </div>
        </footer>
      </div>
    </div>
  );
};

export default AuthLayout;
