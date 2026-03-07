import { useState, useEffect, useRef } from 'react';
import { Link } from 'react-router-dom';
import {
  Menu,
  X,
  ChevronDown,
  Zap,
  BarChart3,
  Search,
  Layers,
  Cpu,
  Target,
  Settings,
  Truck,
  Warehouse,
  Ship,
  GitBranch,
  Building2,
  Play,
  Database,
  FileBarChart,
  BookOpen,
  Code2,
  HelpCircle,
  Users,
  Mail,
  ArrowRight,
} from 'lucide-react';

/* ──────────────────────────────────────────
   Navigation Data
   ────────────────────────────────────────── */

interface NavSubItem {
  label: string;
  href: string;
  icon: React.ReactNode;
  description?: string;
}

interface NavItem {
  label: string;
  href?: string;
  children?: NavSubItem[];
}

const navigation: NavItem[] = [
  {
    label: 'Product',
    children: [
      { label: 'Overview', href: '/product', icon: <Layers className="w-5 h-5" />, description: 'What Vyn does for your supply chain' },
      { label: 'How It Works', href: '/product/how-it-works', icon: <Settings className="w-5 h-5" />, description: 'Upload, detect, optimize — in minutes' },
      { label: 'AI Process Intelligence', href: '/product/ai-intelligence', icon: <Cpu className="w-5 h-5" />, description: 'ML-powered process mining engine' },
      { label: 'Bottleneck Detection', href: '/product/bottleneck-detection', icon: <Target className="w-5 h-5" />, description: 'Pinpoint delays with Isolation Forest' },
      { label: 'Process Analytics', href: '/product/analytics', icon: <BarChart3 className="w-5 h-5" />, description: 'Real-time throughput and risk metrics' },
      { label: 'Technology', href: '/product/technology', icon: <Zap className="w-5 h-5" />, description: 'Our tech stack and architecture' },
    ],
  },
  {
    label: 'Solutions',
    children: [
      { label: 'Trucking & Delivery', href: '/solutions/trucking', icon: <Truck className="w-5 h-5" />, description: 'Optimize last-mile delivery routes' },
      { label: 'Warehouse Fulfillment', href: '/solutions/warehouse', icon: <Warehouse className="w-5 h-5" />, description: 'Reduce pick-pack-ship cycle times' },
      { label: 'Import / Export Logistics', href: '/solutions/import-export', icon: <Ship className="w-5 h-5" />, description: 'Streamline customs and port operations' },
      { label: 'Supply Chain Optimization', href: '/solutions/supply-chain', icon: <GitBranch className="w-5 h-5" />, description: 'End-to-end supply chain visibility' },
      { label: 'Enterprise Monitoring', href: '/solutions/enterprise', icon: <Building2 className="w-5 h-5" />, description: 'Multi-facility, multi-region monitoring' },
    ],
  },
  {
    label: 'Demo',
    children: [
      { label: 'Interactive Demo', href: '/demo', icon: <Play className="w-5 h-5" />, description: 'Try Vyn with live data' },
      { label: 'Sample Dataset', href: '/demo/sample', icon: <Database className="w-5 h-5" />, description: 'Download a sample CSV to explore' },
      { label: 'Example Analysis', href: '/demo/example', icon: <FileBarChart className="w-5 h-5" />, description: 'See a completed bottleneck report' },
    ],
  },
  {
    label: 'Pricing',
    href: '/pricing',
  },
  {
    label: 'Resources',
    children: [
      { label: 'Documentation', href: '/resources/docs', icon: <BookOpen className="w-5 h-5" />, description: 'Guides, tutorials, and references' },
      { label: 'API Reference', href: '/resources/api', icon: <Code2 className="w-5 h-5" />, description: 'RESTful API for integrations' },
      { label: 'Help Center', href: '/resources/help', icon: <HelpCircle className="w-5 h-5" />, description: 'FAQs and support articles' },
    ],
  },
  {
    label: 'About Us',
    children: [
      { label: 'About', href: '/about-us/about', icon: <Users className="w-5 h-5" />, description: 'Our mission and team' },
      { label: 'Contact', href: '/about-us/contact', icon: <Mail className="w-5 h-5" />, description: 'Get in touch with us' },
    ],
  },
];

/* ──────────────────────────────────────────
   Dropdown Component
   ────────────────────────────────────────── */

interface DropdownProps {
  items: NavSubItem[];
  isOpen: boolean;
  onClose: () => void;
}

const Dropdown = ({ items, isOpen, onClose }: DropdownProps) => {
  if (!isOpen) return null;

  return (
    <div
      className="absolute top-full left-1/2 -translate-x-1/2 pt-3 z-50"
      onMouseLeave={onClose}
    >
      <div className="bg-white rounded-xl shadow-dropdown border border-border p-2 min-w-[320px] animate-slide-down">
        {items.map((item) => (
          <Link
            key={item.href}
            to={item.href}
            onClick={onClose}
            className="flex items-start gap-3 px-3 py-2.5 rounded-lg hover:bg-surface transition-colors duration-150 group"
          >
            <span className="mt-0.5 text-content-muted group-hover:text-orange transition-colors duration-150">
              {item.icon}
            </span>
            <div className="flex-1 min-w-0">
              <p className="text-sm font-medium text-content-primary group-hover:text-navy transition-colors duration-150">
                {item.label}
              </p>
              {item.description && (
                <p className="text-xs text-content-muted mt-0.5 leading-relaxed">
                  {item.description}
                </p>
              )}
            </div>
          </Link>
        ))}
      </div>
    </div>
  );
};

/* ──────────────────────────────────────────
   Mobile Menu
   ────────────────────────────────────────── */

interface MobileMenuProps {
  isOpen: boolean;
  onClose: () => void;
}

const MobileMenu = ({ isOpen, onClose }: MobileMenuProps) => {
  const [expandedItem, setExpandedItem] = useState<string | null>(null);

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 lg:hidden">
      {/* Backdrop */}
      <div className="fixed inset-0 bg-black/20 backdrop-blur-sm" onClick={onClose} />

      {/* Panel */}
      <div className="fixed top-0 right-0 w-full max-w-sm h-full bg-white shadow-2xl animate-fade-in overflow-y-auto">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-border">
          <Link to="/" onClick={onClose} className="flex items-center gap-2">
            <div className="w-8 h-8 bg-navy rounded-lg flex items-center justify-center">
              <Zap className="w-4 h-4 text-orange" />
            </div>
            <span className="text-lg font-bold text-navy">Vyn</span>
          </Link>
          <button
            onClick={onClose}
            className="p-2 rounded-lg hover:bg-surface transition-colors"
            aria-label="Close menu"
          >
            <X className="w-5 h-5 text-content-secondary" />
          </button>
        </div>

        {/* Nav Items */}
        <nav className="px-4 py-4">
          {navigation.map((item) => (
            <div key={item.label} className="mb-1">
              {item.children ? (
                <>
                  <button
                    onClick={() => setExpandedItem(expandedItem === item.label ? null : item.label)}
                    className="flex items-center justify-between w-full px-3 py-2.5 rounded-lg text-sm font-medium text-content-primary hover:bg-surface transition-colors"
                  >
                    {item.label}
                    <ChevronDown
                      className={`w-4 h-4 text-content-muted transition-transform duration-200 ${
                        expandedItem === item.label ? 'rotate-180' : ''
                      }`}
                    />
                  </button>
                  {expandedItem === item.label && (
                    <div className="ml-3 pl-3 border-l-2 border-orange/30 mt-1 mb-2 animate-fade-in">
                      {item.children.map((child) => (
                        <Link
                          key={child.href}
                          to={child.href}
                          onClick={onClose}
                          className="flex items-center gap-3 px-3 py-2 rounded-lg text-sm text-content-secondary hover:text-navy hover:bg-surface transition-colors"
                        >
                          <span className="text-content-muted">{child.icon}</span>
                          {child.label}
                        </Link>
                      ))}
                    </div>
                  )}
                </>
              ) : (
                <Link
                  to={item.href!}
                  onClick={onClose}
                  className="block px-3 py-2.5 rounded-lg text-sm font-medium text-content-primary hover:bg-surface transition-colors"
                >
                  {item.label}
                </Link>
              )}
            </div>
          ))}
        </nav>

        {/* Auth Actions */}
        <div className="px-4 py-4 border-t border-border mt-2">
          <Link
            to="/login"
            onClick={onClose}
            className="block w-full text-center px-4 py-2.5 mb-2 rounded-lg text-sm font-medium text-navy hover:bg-surface transition-colors"
          >
            Sign In
          </Link>
          <Link
            to="/register"
            onClick={onClose}
            className="flex items-center justify-center gap-2 w-full px-4 py-2.5 rounded-lg text-sm font-semibold text-white bg-orange hover:bg-orange-dark transition-colors shadow-sm"
          >
            Get Started Free
            <ArrowRight className="w-4 h-4" />
          </Link>
        </div>
      </div>
    </div>
  );
};

/* ──────────────────────────────────────────
   Main Navbar Component
   ────────────────────────────────────────── */

const Navbar = () => {
  const [openDropdown, setOpenDropdown] = useState<string | null>(null);
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [scrolled, setScrolled] = useState(false);
  const timeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Track scroll for shadow effect
  useEffect(() => {
    const handleScroll = () => setScrolled(window.scrollY > 10);
    window.addEventListener('scroll', handleScroll, { passive: true });
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  // Close mobile menu on resize
  useEffect(() => {
    const handleResize = () => {
      if (window.innerWidth >= 1024) setMobileMenuOpen(false);
    };
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  const handleMouseEnter = (label: string) => {
    if (timeoutRef.current) clearTimeout(timeoutRef.current);
    setOpenDropdown(label);
  };

  const handleMouseLeave = () => {
    timeoutRef.current = setTimeout(() => setOpenDropdown(null), 150);
  };

  return (
    <>
      <header
        className={`fixed top-0 left-0 right-0 z-40 bg-white/95 backdrop-blur-md transition-shadow duration-300 ${
          scrolled ? 'shadow-navbar border-b border-border/50' : ''
        }`}
      >
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16 lg:h-[68px]">

            {/* Logo */}
            <Link to="/" className="flex items-center gap-2.5 shrink-0">
              <img src="/logo.svg" alt="Vyn Logo" className="w-9 h-9" />
              <div className="flex flex-col">
                <span className="text-lg font-bold text-navy leading-tight tracking-tight">
                  Vyn
                </span>
                <span className="text-[10px] font-medium text-content-muted leading-none tracking-wider uppercase hidden sm:block">
                  Logistics Intelligence
                </span>
              </div>
            </Link>

            {/* Desktop Navigation */}
            <nav className="hidden lg:flex items-center gap-1">
              {navigation.map((item) => (
                <div
                  key={item.label}
                  className="relative"
                  onMouseEnter={() => item.children && handleMouseEnter(item.label)}
                  onMouseLeave={handleMouseLeave}
                >
                  {item.children ? (
                    <button
                      className={`flex items-center gap-1 px-3 py-2 rounded-lg text-sm font-medium transition-colors duration-150 ${
                        openDropdown === item.label
                          ? 'text-navy bg-surface'
                          : 'text-content-secondary hover:text-navy hover:bg-surface/60'
                      }`}
                    >
                      {item.label}
                      <ChevronDown
                        className={`w-3.5 h-3.5 transition-transform duration-200 ${
                          openDropdown === item.label ? 'rotate-180' : ''
                        }`}
                      />
                    </button>
                  ) : (
                    <Link
                      to={item.href!}
                      className="flex items-center px-3 py-2 rounded-lg text-sm font-medium text-content-secondary hover:text-navy hover:bg-surface/60 transition-colors duration-150"
                    >
                      {item.label}
                    </Link>
                  )}

                  {/* Dropdown */}
                  {item.children && (
                    <Dropdown
                      items={item.children}
                      isOpen={openDropdown === item.label}
                      onClose={() => setOpenDropdown(null)}
                    />
                  )}
                </div>
              ))}
            </nav>

            {/* Right side: Search + Auth */}
            <div className="flex items-center gap-2">
              {/* Search (desktop) */}
              <button
                className="hidden lg:flex items-center gap-2 px-3 py-1.5 rounded-lg border border-border text-sm text-content-muted hover:border-border-dark hover:text-content-secondary transition-colors"
                aria-label="Search"
              >
                <Search className="w-4 h-4" />
                <span className="text-xs">Search...</span>
                <kbd className="hidden xl:inline-flex items-center px-1.5 py-0.5 rounded border border-border bg-surface text-[10px] font-mono text-content-muted">
                  ⌘K
                </kbd>
              </button>

              {/* Sign In */}
              <Link
                to="/login"
                className="hidden lg:flex items-center px-4 py-2 rounded-lg text-sm font-medium text-navy hover:bg-surface transition-colors"
              >
                Sign In
              </Link>

              {/* CTA — Get Started */}
              <Link
                to="/register"
                className="hidden sm:flex items-center gap-1.5 px-4 py-2 rounded-lg text-sm font-semibold text-white bg-orange hover:bg-orange-dark active:scale-[0.97] transition-all duration-150 shadow-sm hover:shadow-md"
              >
                Get Started
                <ArrowRight className="w-3.5 h-3.5" />
              </Link>

              {/* Mobile Hamburger */}
              <button
                onClick={() => setMobileMenuOpen(true)}
                className="lg:hidden p-2 rounded-lg hover:bg-surface transition-colors"
                aria-label="Open menu"
              >
                <Menu className="w-5 h-5 text-content-primary" />
              </button>
            </div>

          </div>
        </div>
      </header>

      {/* Mobile Menu */}
      <MobileMenu isOpen={mobileMenuOpen} onClose={() => setMobileMenuOpen(false)} />
    </>
  );
};

export default Navbar;
