import { useState } from 'react';
import { Link } from 'react-router-dom';
import {
  Mail,
  Lock,
  Eye,
  EyeOff,
  ArrowRight,
  Github,
} from 'lucide-react';

/* ──────────────────────────────────────────
   Google Icon (inline SVG for brand accuracy)
   ────────────────────────────────────────── */

const GoogleIcon = () => (
  <svg className="w-5 h-5" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
    <path d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92a5.06 5.06 0 0 1-2.2 3.32v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.1z" fill="#4285F4" />
    <path d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z" fill="#34A853" />
    <path d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z" fill="#FBBC05" />
    <path d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z" fill="#EA4335" />
  </svg>
);

/* ──────────────────────────────────────────
   Login Page Component
   ────────────────────────────────────────── */

const LoginPage = () => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [rememberMe, setRememberMe] = useState(false);
  const [submitted, setSubmitted] = useState(false);
  const [touched, setTouched] = useState<Record<string, boolean>>({});

  const showError = (field: string, value: string) =>
    (touched[field] || submitted) && value.trim() === '';

  const inputClass = (field: string, value: string, extra = '') =>
    `w-full pl-10 pr-4 py-2.5 bg-white border rounded-xl text-sm text-content-primary placeholder-content-muted focus:outline-none focus:ring-2 transition-all ${extra} ${
      showError(field, value)
        ? 'border-danger focus:ring-danger/40 focus:border-danger bg-danger-50/30'
        : 'border-border focus:ring-orange/40 focus:border-orange'
    }`;

  const handleBlur = (field: string) =>
    setTouched((prev) => ({ ...prev, [field]: true }));

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setSubmitted(true);
    if (!email.trim() || !password.trim()) return;
    // TODO: Integrate with auth API
    console.log('Login attempt:', { email, rememberMe });
  };

  return (
    <div>
      {/* Header */}
      <div className="mb-8">
        <h2 className="text-2xl font-bold text-content-primary mb-2">
          Welcome back
        </h2>
        <p className="text-sm text-content-secondary">
          Sign in to your Vyn account to continue
        </p>
      </div>

      {/* Social Login Buttons */}
      <div className="flex gap-3 mb-6">
        <button
          type="button"
          className="flex-1 flex items-center justify-center gap-2.5 px-4 py-2.5 bg-white border border-border rounded-xl text-sm font-medium text-content-primary hover:bg-surface hover:border-border-dark transition-all duration-150 active:scale-[0.98]"
        >
          <GoogleIcon />
          Google
        </button>
        <button
          type="button"
          className="flex-1 flex items-center justify-center gap-2.5 px-4 py-2.5 bg-white border border-border rounded-xl text-sm font-medium text-content-primary hover:bg-surface hover:border-border-dark transition-all duration-150 active:scale-[0.98]"
        >
          <Github className="w-5 h-5" />
          GitHub
        </button>
      </div>

      {/* Divider */}
      <div className="relative mb-6">
        <div className="absolute inset-0 flex items-center">
          <div className="w-full border-t border-border" />
        </div>
        <div className="relative flex justify-center text-xs">
          <span className="bg-white px-3 text-content-muted font-medium uppercase tracking-wider">
            or continue with email
          </span>
        </div>
      </div>

      {/* Login Form */}
      <form onSubmit={handleSubmit} className="space-y-4">
        {/* Email */}
        <div>
          <label htmlFor="login-email" className="block text-sm font-medium text-content-primary mb-1.5">
            Email address
          </label>
          <div className="relative">
            <Mail className={`absolute left-3.5 top-1/2 -translate-y-1/2 w-4 h-4 pointer-events-none ${showError('email', email) ? 'text-danger' : 'text-content-muted'}`} />
            <input
              id="login-email"
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              onBlur={() => handleBlur('email')}
              placeholder="you@company.com"
              autoComplete="email"
              className={inputClass('email', email)}
            />
          </div>
          {showError('email', email) && (
            <p className="text-xs text-danger mt-1.5">Email is required</p>
          )}
        </div>

        {/* Password */}
        <div>
          <label htmlFor="login-password" className="block text-sm font-medium text-content-primary mb-1.5">
            Password
          </label>
          <div className="relative">
            <Lock className={`absolute left-3.5 top-1/2 -translate-y-1/2 w-4 h-4 pointer-events-none ${showError('password', password) ? 'text-danger' : 'text-content-muted'}`} />
            <input
              id="login-password"
              type={showPassword ? 'text' : 'password'}
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              onBlur={() => handleBlur('password')}
              placeholder="Enter your password"
              autoComplete="current-password"
              className={inputClass('password', password, 'pr-11')}
            />
            <button
              type="button"
              onClick={() => setShowPassword(!showPassword)}
              className="absolute right-3 top-1/2 -translate-y-1/2 p-0.5 text-content-muted hover:text-content-secondary transition-colors"
              aria-label={showPassword ? 'Hide password' : 'Show password'}
            >
              {showPassword ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
            </button>
          </div>
          {showError('password', password) && (
            <p className="text-xs text-danger mt-1.5">Password is required</p>
          )}
        </div>

        {/* Remember + Forgot */}
        <div className="flex items-center justify-between">
          <label className="flex items-center gap-2 cursor-pointer select-none">
            <input
              type="checkbox"
              checked={rememberMe}
              onChange={(e) => setRememberMe(e.target.checked)}
              className="w-4 h-4 rounded border-border text-orange focus:ring-orange/40 focus:ring-2 cursor-pointer accent-orange"
            />
            <span className="text-sm text-content-secondary">Remember me</span>
          </label>
          <Link
            to="/forgot-password"
            className="text-sm font-medium text-orange hover:text-orange-dark transition-colors"
          >
            Forgot password?
          </Link>
        </div>

        {/* Submit */}
        <button
          type="submit"
          className="w-full flex items-center justify-center gap-2 px-6 py-3 bg-orange hover:bg-orange-dark rounded-xl text-base font-semibold text-white transition-all shadow-md hover:shadow-lg hover:shadow-orange/20 active:scale-[0.98] animate-pulse-glow"
        >
          Sign In
          <ArrowRight className="w-4 h-4" />
        </button>
      </form>

      {/* Bottom Link */}
      <p className="text-center text-sm text-content-secondary mt-8">
        Don't have an account?{' '}
        <Link
          to="/register"
          className="font-semibold text-orange hover:text-orange-dark transition-colors"
        >
          Create one
        </Link>
      </p>
    </div>
  );
};

export default LoginPage;
