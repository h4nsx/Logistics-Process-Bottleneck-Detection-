import { useState, useEffect, useCallback } from 'react';
import { Link } from 'react-router-dom';
import {
  Mail,
  ArrowRight,
  ArrowLeft,
  CheckCircle2,
  RefreshCw,
} from 'lucide-react';

/* ──────────────────────────────────────────
   Forgot Password Page Component
   ────────────────────────────────────────── */

const ForgotPasswordPage = () => {
  const [email, setEmail] = useState('');
  const [submitted, setSubmitted] = useState(false);
  const [touched, setTouched] = useState(false);
  const [sent, setSent] = useState(false);
  const [resendCooldown, setResendCooldown] = useState(0);

  const showError = (touched || submitted) && email.trim() === '';

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setSubmitted(true);
    if (!email.trim()) return;
    // TODO: Integrate with auth API
    console.log('Password reset requested for:', email);
    setSent(true);
    setResendCooldown(60);
  };

  const handleResend = useCallback(() => {
    if (resendCooldown > 0) return;
    console.log('Resending reset email to:', email);
    setResendCooldown(60);
  }, [resendCooldown, email]);

  // Cooldown timer
  useEffect(() => {
    if (resendCooldown <= 0) return;
    const timer = setInterval(() => {
      setResendCooldown((prev) => (prev <= 1 ? 0 : prev - 1));
    }, 1000);
    return () => clearInterval(timer);
  }, [resendCooldown]);

  /* ── Success State ─────────────────────── */
  if (sent) {
    return (
      <div className="text-center">
        {/* Animated checkmark */}
        <div className="w-16 h-16 rounded-full bg-success-50 flex items-center justify-center mx-auto mb-6 ring-4 ring-success/10">
          <CheckCircle2 className="w-8 h-8 text-success" />
        </div>

        <h2 className="text-2xl font-bold text-content-primary mb-2">
          Check your inbox
        </h2>
        <p className="text-sm text-content-secondary mb-2 leading-relaxed">
          We've sent a password reset link to
        </p>
        <p className="text-sm font-semibold text-content-primary mb-8">
          {email}
        </p>

        {/* Resend button */}
        <button
          type="button"
          onClick={handleResend}
          disabled={resendCooldown > 0}
          className="inline-flex items-center gap-2 px-4 py-2 text-sm font-medium text-orange hover:text-orange-dark disabled:text-content-muted transition-colors"
        >
          <RefreshCw className={`w-4 h-4 ${resendCooldown > 0 ? '' : 'group-hover:rotate-180'}`} />
          {resendCooldown > 0
            ? `Resend in ${resendCooldown}s`
            : 'Resend email'}
        </button>

        {/* Divider */}
        <div className="my-6 border-t border-border" />

        {/* Back to login */}
        <Link
          to="/login"
          className="inline-flex items-center gap-2 text-sm font-medium text-content-secondary hover:text-content-primary transition-colors"
        >
          <ArrowLeft className="w-4 h-4" />
          Back to sign in
        </Link>
      </div>
    );
  }

  /* ── Email Form State ──────────────────── */
  return (
    <div>
      {/* Header */}
      <div className="mb-8">
        <h2 className="text-2xl font-bold text-content-primary mb-2">
          Reset your password
        </h2>
        <p className="text-sm text-content-secondary leading-relaxed">
          Enter the email address associated with your account and we'll send you a link to reset your password.
        </p>
      </div>

      {/* Form */}
      <form onSubmit={handleSubmit} className="space-y-4">
        {/* Email */}
        <div>
          <label htmlFor="reset-email" className="block text-sm font-medium text-content-primary mb-1.5">
            Email address
          </label>
          <div className="relative">
            <Mail className={`absolute left-3.5 top-1/2 -translate-y-1/2 w-4 h-4 pointer-events-none ${showError ? 'text-danger' : 'text-content-muted'}`} />
            <input
              id="reset-email"
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              onBlur={() => setTouched(true)}
              placeholder="your@email.com"
              autoComplete="email"
              autoFocus
              className={`w-full pl-10 pr-4 py-2.5 bg-white border rounded-xl text-sm text-content-primary placeholder-content-muted focus:outline-none focus:ring-2 transition-all ${
                showError
                  ? 'border-danger focus:ring-danger/40 focus:border-danger bg-danger-50/30'
                  : 'border-border focus:ring-orange/40 focus:border-orange'
              }`}
            />
          </div>
          {showError && (
            <p className="text-xs text-danger mt-1.5">Email is required</p>
          )}
        </div>

        {/* Submit */}
        <button
          type="submit"
          className="w-full flex items-center justify-center gap-2 px-6 py-3 bg-orange hover:bg-orange-dark rounded-xl text-base font-semibold text-white transition-all shadow-md hover:shadow-lg hover:shadow-orange/20 active:scale-[0.98]"
        >
          Send Reset Link
          <ArrowRight className="w-4 h-4" />
        </button>
      </form>

      {/* Back to login */}
      <div className="text-center mt-8">
        <Link
          to="/login"
          className="inline-flex items-center gap-2 text-sm font-medium text-content-secondary hover:text-content-primary transition-colors"
        >
          <ArrowLeft className="w-4 h-4" />
          Back to sign in
        </Link>
      </div>
    </div>
  );
};

export default ForgotPasswordPage;
