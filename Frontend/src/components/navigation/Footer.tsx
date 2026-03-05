import { Link } from 'react-router-dom';
import {
  Zap,
  Github,
  ArrowRight,
  Mail,
} from 'lucide-react';

/* ──────────────────────────────────────────
   Footer Navigation Data
   ────────────────────────────────────────── */

interface FooterLink {
  label: string;
  href: string;
}

interface FooterColumn {
  title: string;
  links: FooterLink[];
}

const footerColumns: FooterColumn[] = [
  {
    title: 'Product',
    links: [
      { label: 'Overview', href: '/product' },
      { label: 'How It Works', href: '/product/how-it-works' },
      { label: 'AI Process Intelligence', href: '/product/ai-intelligence' },
      { label: 'Bottleneck Detection', href: '/product/bottleneck-detection' },
      { label: 'Process Analytics', href: '/product/analytics' },
      { label: 'Technology', href: '/product/technology' },
    ],
  },
  {
    title: 'Solutions',
    links: [
      { label: 'Trucking & Delivery', href: '/solutions/trucking' },
      { label: 'Warehouse Fulfillment', href: '/solutions/warehouse' },
      { label: 'Import / Export', href: '/solutions/import-export' },
      { label: 'Supply Chain', href: '/solutions/supply-chain' },
      { label: 'Enterprise Monitoring', href: '/solutions/enterprise' },
    ],
  },
  {
    title: 'Resources',
    links: [
      { label: 'Documentation', href: '/resources/docs' },
      { label: 'API Reference', href: '/resources/api' },
      { label: 'Help Center', href: '/resources/help' },
      { label: 'Interactive Demo', href: '/demo' },
    ],
  },
  {
    title: 'About Us',
    links: [
      { label: 'About', href: '/company/about' },
      { label: 'Contact', href: '/company/contact' },
      { label: 'Careers', href: '/company/careers' },
      { label: 'Pricing', href: '/pricing' },
    ],
  },
];

const socialLinks = [
  { label: 'GitHub', href: 'https://github.com/h4nsx/Logistics-Process-Bottleneck-Detection-', icon: <Github className="w-5 h-5" /> },
];

/* ──────────────────────────────────────────
   Newsletter Signup
   ────────────────────────────────────────── */

const NewsletterForm = () => {
  return (
    <div>
      <h3 className="text-sm font-semibold text-white mb-3">
        Stay Updated
      </h3>
      <p className="text-sm text-slate-400 mb-4 leading-relaxed">
        Get the latest on process mining, logistics AI, and product updates.
      </p>
      <form
        onSubmit={(e) => e.preventDefault()}
        className="flex gap-2"
      >
        <div className="relative flex-1">
          <Mail className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-500" />
          <input
            type="email"
            placeholder="you@company.com"
            className="w-full pl-9 pr-3 py-2.5 bg-slate-800 border border-slate-700 rounded-lg text-sm text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-orange/50 focus:border-orange transition-all"
          />
        </div>
        <button
          type="submit"
          className="flex items-center gap-1.5 px-4 py-2.5 bg-orange hover:bg-orange-dark rounded-lg text-sm font-semibold text-white transition-colors shrink-0 shadow-sm hover:shadow-md"
        >
          Subscribe
          <ArrowRight className="w-3.5 h-3.5" />
        </button>
      </form>
      <p className="text-xs text-slate-500 mt-2">
        No spam, unsubscribe anytime.
      </p>
    </div>
  );
};

/* ──────────────────────────────────────────
   Main Footer Component
   ────────────────────────────────────────── */

const Footer = () => {
  const currentYear = new Date().getFullYear();

  return (
    <footer className="bg-navy-900 border-t border-slate-800">

      {/* ── CTA Banner ─────────────────────── */}
      <div className="border-b border-slate-800">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12 lg:py-16">
          <div className="flex flex-col lg:flex-row items-center justify-between gap-6">
            <div className="text-center lg:text-left">
              <h2 className="text-2xl lg:text-3xl font-bold text-white mb-2">
                Ready to detect bottlenecks faster?
              </h2>
              <p className="text-slate-400 text-base lg:text-lg max-w-xl">
                Upload your first dataset and get actionable insights in under 5 minutes.
              </p>
            </div>
            <div className="flex flex-col sm:flex-row items-center gap-3">
              <Link
                to="/register"
                className="flex items-center gap-2 px-6 py-3 bg-orange hover:bg-orange-dark rounded-xl text-base font-semibold text-white transition-all shadow-lg hover:shadow-xl hover:shadow-orange/20 active:scale-[0.97]"
              >
                Start Free Trial
                <ArrowRight className="w-4 h-4" />
              </Link>
              <Link
                to="/demo"
                className="flex items-center gap-2 px-6 py-3 border border-slate-600 hover:border-slate-500 rounded-xl text-base font-medium text-slate-300 hover:text-white transition-colors"
              >
                View Live Demo
              </Link>
            </div>
          </div>
        </div>
      </div>

      {/* ── Main Footer Grid ──────────────── */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12 lg:py-16">
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-8 lg:gap-12">

          {/* Brand + Newsletter Column (spans 2 cols on large) */}
          <div className="col-span-2">
            {/* Logo */}
            <Link to="/" className="flex items-center gap-2.5 mb-5">
              <div className="w-9 h-9 bg-navy rounded-xl flex items-center justify-center ring-1 ring-slate-700">
                <Zap className="w-5 h-5 text-orange" />
              </div>
              <div className="flex flex-col">
                <span className="text-lg font-bold text-white leading-tight tracking-tight">
                  Vyn
                </span>
                <span className="text-[10px] font-medium text-slate-500 leading-none tracking-wider uppercase">
                  Logistics Intelligence
                </span>
              </div>
            </Link>

            <p className="text-sm text-slate-400 leading-relaxed mb-6 max-w-xs">
              Fast anomaly detection and logistics intelligence. 
              Identify bottlenecks, optimize throughput, and accelerate your supply chain with AI-powered process mining.
            </p>

            {/* Newsletter */}
            <NewsletterForm />
          </div>

          {/* Navigation Columns */}
          {footerColumns.map((column) => (
            <div key={column.title}>
              <h3 className="text-sm font-semibold text-white mb-4">
                {column.title}
              </h3>
              <ul className="space-y-2.5">
                {column.links.map((link) => (
                  <li key={link.href}>
                    <Link
                      to={link.href}
                      className="text-sm text-slate-400 hover:text-orange transition-colors duration-150"
                    >
                      {link.label}
                    </Link>
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>
      </div>

      {/* ── Bottom Bar ─────────────────────── */}
      <div className="border-t border-slate-800">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="flex flex-col md:flex-row items-center justify-between gap-4">
            {/* Copyright */}
            <div className="flex flex-wrap items-center justify-center md:justify-start gap-x-4 gap-y-1 text-sm text-slate-500">
              <span>© {currentYear} Vyn. All rights reserved.</span>
              <span className="hidden md:inline text-slate-700">·</span>
              <Link to="/privacy" className="hover:text-slate-300 transition-colors">
                Privacy Policy
              </Link>
              <span className="hidden md:inline text-slate-700">·</span>
              <Link to="/terms" className="hover:text-slate-300 transition-colors">
                Terms of Service
              </Link>
              <span className="hidden md:inline text-slate-700">·</span>
              <Link to="/cookies" className="hover:text-slate-300 transition-colors">
                Cookie Policy
              </Link>
            </div>

            {/* Social Links */}
            <div className="flex items-center gap-1">
              {socialLinks.map((social) => (
                <a
                  key={social.label}
                  href={social.href}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="p-2 rounded-lg text-slate-500 hover:text-white hover:bg-slate-800 transition-colors duration-150"
                  aria-label={social.label}
                >
                  {social.icon}
                </a>
              ))}
            </div>
          </div>
        </div>
      </div>

    </footer>
  );
};

export default Footer;
