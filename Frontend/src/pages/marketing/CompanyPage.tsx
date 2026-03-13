import { Link } from 'react-router-dom';
import {
  Target,
  Zap,
  Users,
  Globe,
  ArrowRight,
  TrendingUp,
  Lightbulb,
  Shield,
  Heart,
  Github,
  Facebook,
  Mail,
} from 'lucide-react';

/* ──────────────────────────────────────────
   Section 1: Hero
   ────────────────────────────────────────── */

const HeroSection = () => (
  <section className="relative overflow-hidden bg-gradient-to-b from-navy-50 via-white to-white py-20 lg:py-28">
    {/* Background decoration — animated orbs */}
    <div className="absolute top-0 left-1/2 -translate-x-1/2 w-[800px] h-[400px] bg-gradient-to-b from-orange/5 to-transparent rounded-full blur-3xl animate-pulse" style={{ animationDuration: '4s' }} />
    <div className="absolute -top-20 -left-20 w-72 h-72 bg-cyan/5 rounded-full blur-3xl animate-float" />
    <div className="absolute -bottom-10 -right-20 w-64 h-64 bg-orange/5 rounded-full blur-3xl animate-float-delayed" />

    <div className="absolute inset-0 opacity-[0.03]" style={{
      backgroundImage: 'radial-gradient(circle at 1px 1px, currentColor 1px, transparent 0)',
      backgroundSize: '28px 28px',
    }} />

    <div className="relative max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
      <div
        className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-orange-50 border border-orange-100 mb-6 opacity-0 animate-[fadeInUp_0.6s_ease-out_0.2s_forwards]"
      >
        <Globe className="w-3.5 h-3.5 text-orange" />
        <span className="text-xs font-semibold text-orange-dark tracking-wide uppercase">About Vyn</span>
      </div>

      {/* Headline — slides up second */}
      <h1
        className="text-4xl lg:text-5xl font-bold text-content-primary mb-6 leading-tight opacity-0 animate-[fadeInUp_0.7s_ease-out_0.4s_forwards]"
      >
        Shining a light on
        <br />
        <span className="text-transparent bg-clip-text bg-gradient-to-r from-orange via-orange-dark to-orange animate-shimmer bg-[length:200%_auto]">
          hidden logistics bottlenecks
        </span>
      </h1>

      {/* Subtitle — third */}
      <p
        className="text-lg text-content-secondary leading-relaxed max-w-2xl mx-auto mb-10 opacity-0 animate-[fadeInUp_0.7s_ease-out_0.6s_forwards]"
      >
        Vyn is a <strong className="text-content-primary">statistical intelligence platform</strong> that helps supply chain teams
        identify which steps are causing abnormal delays — using pure data science, no black-box AI.
      </p>

      {/* Info badges — last */}
      <div
        className="flex flex-wrap items-center justify-center gap-x-8 gap-y-3 text-sm text-content-muted opacity-0 animate-[fadeInUp_0.7s_ease-out_0.8s_forwards]"
      >
        <span className="flex items-center gap-2">
          <Zap className="w-4 h-4 text-orange" />
          Founded 2026
        </span>
        <span className="flex items-center gap-2">
          <Globe className="w-4 h-4 text-cyan" />
          Remote-first
        </span>
        <span className="flex items-center gap-2">
          <Users className="w-4 h-4 text-navy" />
          Open Source
        </span>
      </div>
    </div>
  </section>
);

/* ──────────────────────────────────────────
   Section 2: Mission
   ────────────────────────────────────────── */

const MissionSection = () => (
  <section className="py-16 lg:py-24 bg-white">
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-12 lg:gap-16 items-center">
        {/* Left: Text */}
        <div>
          <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-navy-50 border border-navy-100 mb-4">
            <Target className="w-3.5 h-3.5 text-navy" />
            <span className="text-xs font-semibold text-navy tracking-wide uppercase">Our Mission</span>
          </div>
          <h2 className="text-3xl lg:text-4xl font-bold text-content-primary mb-6 leading-tight">
            Your operations have hidden delays. Vyn tells you exactly where.
          </h2>
          <p className="text-base text-content-secondary leading-relaxed mb-4">
            Logistics teams know delays exist — they feel them in missed SLAs, angry customers, and overloaded staff.
            But pinpointing <em>which</em> step is abnormally slow, and <em>how risky</em> it is, has always required
            costly data science resources.
          </p>
          <p className="text-base text-content-secondary leading-relaxed mb-4">
            Vyn solves this with <strong className="text-content-primary">pure statistical intelligence</strong>:
            we compute the mean, standard deviation, and 95th percentile (p95) for every step in your process —
            then flag executions where the z-score ≥ 2 or duration exceeds p95. No black-box AI.
            No need to expose sensitive business logic.
          </p>
          <p className="text-base text-content-secondary leading-relaxed">
            Just upload your execution timestamps as a CSV, and Vyn surfaces every bottleneck with
            a transparent, auditable risk score — in minutes.
          </p>
        </div>

        {/* Right: Stats Grid */}
        <div className="grid grid-cols-2 gap-4">
          {[
            { value: '< 5 min', label: 'Time to first insight', icon: <Zap className="w-5 h-5" />, color: 'text-orange', bg: 'bg-orange-50' },
            { value: '94%', label: 'Detection accuracy', icon: <Target className="w-5 h-5" />, color: 'text-cyan', bg: 'bg-cyan-50' },
            { value: '40%', label: 'Avg. delay reduction', icon: <TrendingUp className="w-5 h-5" />, color: 'text-success', bg: 'bg-success-50' },
            { value: '3.2x', label: 'Faster resolution', icon: <Shield className="w-5 h-5" />, color: 'text-navy', bg: 'bg-navy-50' },
          ].map((stat) => (
            <div key={stat.label} className="bg-white rounded-2xl border border-border p-6 text-center hover:shadow-elevated transition-all duration-300">
              <div className={`w-10 h-10 rounded-xl ${stat.bg} ${stat.color} flex items-center justify-center mx-auto mb-3`}>
                {stat.icon}
              </div>
              <p className="text-2xl font-extrabold text-content-primary mb-1">{stat.value}</p>
              <p className="text-xs text-content-muted">{stat.label}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  </section>
);

/* ──────────────────────────────────────────
   Section 3: Values
   ────────────────────────────────────────── */

const values = [
  {
    icon: <Zap className="w-6 h-6" />,
    title: 'Speed First',
    description: 'Logistics moves fast. Our platform delivers insights in minutes, not days. Every feature is designed for rapid time-to-value.',
    color: 'text-orange',
    bg: 'bg-orange-50',
  },
  {
    icon: <Lightbulb className="w-6 h-6" />,
    title: 'Statistics, Not Black-Box AI',
    description: 'No machine-learning predictions or opaque models. Vyn uses mean, standard deviation, and p95 baselines — methods your team can audit, explain, and trust.',
    color: 'text-cyan',
    bg: 'bg-cyan-50',
  },
  {
    icon: <Shield className="w-6 h-6" />,
    title: 'No Business Secrets Required',
    description: 'You only share execution timestamps — start_time, end_time, step code, and location. Vyn never asks for pricing, contracts, or any sensitive business data.',
    color: 'text-navy',
    bg: 'bg-navy-50',
  },
  {
    icon: <Heart className="w-6 h-6" />,
    title: 'Open by Default',
    description: 'Built as an open-source project. Transparency in code, community-driven roadmap, and no vendor lock-in.',
    color: 'text-danger',
    bg: 'bg-danger-50',
  },
];

const ValuesSection = () => (
  <section className="py-16 lg:py-24 bg-surface">
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
      <div className="text-center max-w-2xl mx-auto mb-12 lg:mb-16">
        <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-orange-50 border border-orange-100 mb-4">
          <Heart className="w-3.5 h-3.5 text-orange" />
          <span className="text-xs font-semibold text-orange-dark tracking-wide uppercase">Our Values</span>
        </div>
        <h2 className="text-3xl lg:text-4xl font-bold text-content-primary mb-4">
          What drives us
        </h2>
        <p className="text-lg text-content-secondary leading-relaxed">
          The principles behind every line of code and every product decision.
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 max-w-4xl mx-auto">
        {values.map((value) => (
          <div
            key={value.title}
            className="bg-white rounded-2xl border border-border p-6 hover:shadow-elevated transition-all duration-300"
          >
            <div className={`w-12 h-12 rounded-xl ${value.bg} ${value.color} flex items-center justify-center mb-4`}>
              {value.icon}
            </div>
            <h3 className="text-lg font-semibold text-content-primary mb-2">
              {value.title}
            </h3>
            <p className="text-sm text-content-secondary leading-relaxed">
              {value.description}
            </p>
          </div>
        ))}
      </div>
    </div>
  </section>
);

/* ──────────────────────────────────────────
   Section 4: Tech Stack
   ────────────────────────────────────────── */

const TechSection = () => (
  <section className="py-16 lg:py-24 bg-white">
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
      <div className="text-center max-w-2xl mx-auto mb-12 lg:mb-16">
        <h2 className="text-3xl lg:text-4xl font-bold text-content-primary mb-4">
          Built with modern technology
        </h2>
        <p className="text-lg text-content-secondary leading-relaxed">
          A carefully chosen stack for performance, developer experience, and reliability.
        </p>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 max-w-3xl mx-auto">
        {[
          { name: 'React', category: 'Frontend' },
          { name: 'TypeScript', category: 'Language' },
          { name: 'Tailwind CSS', category: 'Styling' },
          { name: 'Vite', category: 'Build Tool' },
          { name: 'Python', category: 'Backend' },
          { name: 'Scikit-learn', category: 'ML Engine' },
          { name: 'React Flow', category: 'Visualization' },
          { name: 'Recharts', category: 'Analytics' },
        ].map((tech) => (
          <div
            key={tech.name}
            className="bg-surface rounded-xl border border-border p-4 text-center hover:border-orange/30 hover:shadow-card transition-all duration-300"
          >
            <p className="text-sm font-semibold text-content-primary mb-0.5">{tech.name}</p>
            <p className="text-xs text-content-muted">{tech.category}</p>
          </div>
        ))}
      </div>
    </div>
  </section>
);

/* ──────────────────────────────────────────
   Section 5: Open Source CTA
   ────────────────────────────────────────── */

const OpenSourceSection = () => (
  <section className="py-16 lg:py-24 bg-surface">
    <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
      <div className="bg-gradient-to-br from-navy to-navy-dark rounded-3xl p-10 lg:p-16 relative overflow-hidden">
        {/* Background decoration */}
        <div className="absolute top-0 right-0 w-64 h-64 bg-orange/10 rounded-full blur-3xl" />
        <div className="absolute bottom-0 left-0 w-48 h-48 bg-cyan/10 rounded-full blur-3xl" />

        <div className="relative">
          <div className="w-14 h-14 rounded-2xl bg-white/10 flex items-center justify-center mx-auto mb-6 ring-1 ring-white/10">
            <Github className="w-7 h-7 text-white" />
          </div>

          <h2 className="text-3xl lg:text-4xl font-bold text-white mb-4">
            Open source & community driven
          </h2>
          <p className="text-lg text-slate-300 mb-8 max-w-xl mx-auto leading-relaxed">
            Vyn is open source. Star us on GitHub, dive into the code,
            report issues, or contribute to the roadmap.
          </p>

          <div className="flex flex-col sm:flex-row items-center justify-center gap-3">
            <a
              href="https://github.com/h4nsx/Logistics-Process-Bottleneck-Detection-"
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-2 px-8 py-4 bg-white rounded-xl text-base font-semibold text-navy hover:bg-slate-50 transition-all shadow-lg active:scale-[0.97] w-full sm:w-auto justify-center"
            >
              <Github className="w-5 h-5" />
              View on GitHub
            </a>
            <Link
              to="/register"
              className="flex items-center gap-2 px-8 py-4 bg-orange hover:bg-orange-dark rounded-xl text-base font-semibold text-white transition-all shadow-lg hover:shadow-xl hover:shadow-orange/30 active:scale-[0.97] w-full sm:w-auto justify-center"
            >
              Get Started Free
              <ArrowRight className="w-4 h-4" />
            </Link>
          </div>
        </div>
      </div>
    </div>
  </section>
);

/* ──────────────────────────────────────────
   Section 6: Contact
   ────────────────────────────────────────── */

const teamMembers = [
  {
    name: 'Võ Tuấn Hùng (Hans)',
    role: ' Team Leader',
    bio: 'Full-stack engineer passionate about AI and logistics optimization. Built the core process mining engine.',
    initials: 'Hans',
    image: '',  // ← paste image URL here
    color: 'bg-orange',
    links: {
      github: 'https://github.com/h4nsx',
      facebook: '#',
      email: 'votuanhung1205.work@gmail.com',
    },
  },
  {
    name: 'Nguyễn Quốc Huy (Quwy)',
    role: 'ML Engineer',
    bio: 'Specialized in anomaly detection and statistical modeling. Designed the Isolation Forest pipeline.',
    initials: 'Quwy',
    image: '',  // ← paste image URL here
    color: 'bg-cyan',
    links: {
      github: '#',
      facebook: '#',
      email: 'sarah@vyn.dev',
    },
  },
  {
    name: 'Nguyễn Tăng Minh Thông',
    role: 'Backend Engineer',
    bio: '',
    initials: 'Stone',
    image: '',  
    color: 'bg-navy',
    links: {
      github: '#',
      facebook: '#',
      email: 'marcus@vyn.dev',
    },
  },
  {
    name: 'Nguyễn Văn Linh',
    role: 'Frontend Support & Product & UX Designer',
    bio: 'Designs intuitive interfaces for complex data workflows. Ensures Vyn is accessible to non-technical users.',
    initials: 'Louis',
    image: '', 
    color: 'bg-anomaly',
    links: {
      github: '#',
      facebook: '#',
      linkedin: '#',
      email: 'priya@vyn.dev',
    },
  },
];

const TeamSection = () => (
  <section className="py-16 lg:py-24 bg-white">
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
      <div className="text-center max-w-2xl mx-auto mb-12 lg:mb-16">
        <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-cyan-50 border border-cyan-100 mb-4">
          <Users className="w-3.5 h-3.5 text-cyan" />
          <span className="text-xs font-semibold text-cyan-dark tracking-wide uppercase">Our Team</span>
        </div>
        <h2 className="text-3xl lg:text-4xl font-bold text-content-primary mb-4">
          Meet the people behind Vyn
        </h2>
        <p className="text-lg text-content-secondary leading-relaxed">
          A small, focused team building the future of logistics intelligence.
        </p>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
        {teamMembers.map((member) => (
          <div
            key={member.name}
            className="group bg-white rounded-2xl border border-border p-6 text-center hover:shadow-elevated hover:border-orange/20 transition-all duration-300"
          >
            {/* Avatar */}
            {member.image ? (
              <img
                src={member.image}
                alt={member.name}
                className="w-16 h-16 rounded-2xl object-cover mx-auto mb-4 shadow-md group-hover:scale-105 transition-transform duration-300"
              />
            ) : (
              <div className={`w-16 h-16 rounded-2xl ${member.color} flex items-center justify-center mx-auto mb-4 text-white text-xl font-bold shadow-md group-hover:scale-105 transition-transform duration-300`}>
                {member.initials}
              </div>
            )}

            {/* Info */}
            <h3 className="text-base font-bold text-content-primary mb-0.5">
              {member.name}
            </h3>
            <p className="text-xs font-medium text-orange mb-3">
              {member.role}
            </p>
            <p className="text-sm text-content-secondary leading-relaxed mb-4">
              {member.bio}
            </p>

            {/* Social Links */}
            <div className="flex items-center justify-center gap-1.5">
              <a
                href={member.links.github}
                target="_blank"
                rel="noopener noreferrer"
                className="p-2 rounded-lg text-content-muted hover:text-navy hover:bg-navy-50 transition-colors"
                aria-label={`${member.name}'s GitHub`}
              >
                <Github className="w-4 h-4" />
              </a>
              <a
                href={member.links.linkedin}
                target="_blank"
                rel="noopener noreferrer"
                className="p-2 rounded-lg text-content-muted hover:text-cyan hover:bg-cyan-50 transition-colors"
                aria-label={`${member.name}'s LinkedIn`}
              >
                <Facebook className="w-4 h-4" />
              </a>
              <a
                href={`mailto:${member.links.email}`}
                className="p-2 rounded-lg text-content-muted hover:text-orange hover:bg-orange-50 transition-colors"
                aria-label={`Email ${member.name}`}
              >
                <Mail className="w-4 h-4" />
              </a>
            </div>
          </div>
        ))}
      </div>
    </div>
  </section>
);

/* ──────────────────────────────────────────
   Company Page (All Sections Combined)
   ────────────────────────────────────────── */

const CompanyPage = () => {
  return (
    <>
      <HeroSection />
      <MissionSection />
      <ValuesSection />
      <TechSection />
      <OpenSourceSection />
      <TeamSection />
    </>
  );
};

export default CompanyPage;
