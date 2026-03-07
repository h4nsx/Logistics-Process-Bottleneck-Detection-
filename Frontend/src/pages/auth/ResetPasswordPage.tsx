import { useState, useMemo } from 'react';
import { Link } from 'react-router-dom';
import {
  Lock,
  Eye,
  EyeOff,
  ArrowRight,
  CheckCircle2,
  ShieldCheck,
} from 'lucide-react';

/* ──────────────────────────────────────────
   Password Strength Calculator
   ────────────────────────────────────────── */

interface PasswordStrength {
  score: 0 | 1 | 2 | 3 | 4;
  label: string;
  color: string;
  bgColor: string;
}

const getPasswordStrength = (password: string): PasswordStrength => {
  if (!password) return { score: 0, label: '', color: '', bgColor: 'bg-border' };

  let score = 0;
  if (password.length >= 6) score++;
  if (password.length >= 10) score++;
  if (/[A-Z]/.test(password) && /[a-z]/.test(password)) score++;
  if (/[0-9]/.test(password)) score++;
  if (/[^A-Za-z0-9]/.test(password)) score++;

  if (score <= 1) return { score: 1, label: 'Weak', color: 'text-danger', bgColor: 'bg-danger' };
  if (score === 2) return { score: 2, label: 'Fair', color: 'text-orange', bgColor: 'bg-orange' };
  if (score === 3) return { score: 3, label: 'Good', color: 'text-cyan', bgColor: 'bg-cyan' };
  return { score: 4, label: 'Strong', color: 'text-success', bgColor: 'bg-success' };
};

/* ──────────────────────────────────────────
   Reset Password Page Component
   ────────────────────────────────────────── */

const ResetPasswordPage = () => {
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [showConfirm, setShowConfirm] = useState(false);
  const [submitted, setSubmitted] = useState(false);
  const [touched, setTouched] = useState<Record<string, boolean>>({});
  const [success, setSuccess] = useState(false);

  const strength = useMemo(() => getPasswordStrength(password), [password]);
  const passwordsMatch = confirmPassword.length === 0 || password === confirmPassword;

  const showError = (field: string, value: string) =>
    (touched[field] || submitted) && value.trim() === '';

  const handleBlur = (field: string) =>
    setTouched((prev) => ({ ...prev, [field]: true }));

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setSubmitted(true);
    if (!password.trim() || !confirmPassword.trim() || !passwordsMatch) return;
    // TODO: Integrate with auth API
    console.log('Password reset completed');
    setSuccess(true);
  };

  /* ── Success State ─────────────────────── */
  if (success) {
    return (
      <div className="text-center">
        {/* Animated shield checkmark */}
        <div className="w-16 h-16 rounded-full bg-success-50 flex items-center justify-center mx-auto mb-6 ring-4 ring-success/10">
          <ShieldCheck className="w-8 h-8 text-success" />
        </div>

        <h2 className="text-2xl font-bold text-content-primary mb-2">
          Password updated
        </h2>
        <p className="text-sm text-content-secondary mb-8 leading-relaxed max-w-xs mx-auto">
          Your password has been successfully reset. You can now sign in with your new password.
        </p>

        <Link
          to="/login"
          className="inline-flex items-center gap-2 px-6 py-3 bg-orange hover:bg-orange-dark rounded-xl text-base font-semibold text-white transition-all shadow-md hover:shadow-lg hover:shadow-orange/20 active:scale-[0.98]"
        >
          Continue to Sign In
          <ArrowRight className="w-4 h-4" />
        </Link>
      </div>
    );
  }

  /* ── Form State ────────────────────────── */
  return (
    <div>
      {/* Header */}
      <div className="mb-8">
        <h2 className="text-2xl font-bold text-content-primary mb-2">
          Set new password
        </h2>
        <p className="text-sm text-content-secondary leading-relaxed">
          Your new password must be at least 6 characters and different from your previous password.
        </p>
      </div>

      {/* Form */}
      <form onSubmit={handleSubmit} className="space-y-4">
        {/* New Password */}
        <div>
          <label htmlFor="reset-password" className="block text-sm font-medium text-content-primary mb-1.5">
            New password
          </label>
          <div className="relative">
            <Lock className={`absolute left-3.5 top-1/2 -translate-y-1/2 w-4 h-4 pointer-events-none ${showError('password', password) ? 'text-danger' : 'text-content-muted'}`} />
            <input
              id="reset-password"
              type={showPassword ? 'text' : 'password'}
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              onBlur={() => handleBlur('password')}
              placeholder="Create a strong password"
              minLength={6}
              autoComplete="new-password"
              autoFocus
              className={`w-full pl-10 pr-11 py-2.5 bg-white border rounded-xl text-sm text-content-primary placeholder-content-muted focus:outline-none focus:ring-2 transition-all ${
                showError('password', password)
                  ? 'border-danger focus:ring-danger/40 focus:border-danger bg-danger-50/30'
                  : 'border-border focus:ring-orange/40 focus:border-orange'
              }`}
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

          {/* Password Strength Indicator */}
          {password.length > 0 && (
            <div className="mt-2 space-y-1.5">
              <div className="flex gap-1.5">
                {[1, 2, 3, 4].map((level) => (
                  <div
                    key={level}
                    className={`h-1.5 flex-1 rounded-full transition-colors duration-300 ${
                      level <= strength.score ? strength.bgColor : 'bg-border'
                    }`}
                  />
                ))}
              </div>
              <p className={`text-xs font-medium ${strength.color}`}>
                {strength.label}
              </p>
            </div>
          )}
        </div>

        {/* Confirm Password */}
        <div>
          <label htmlFor="reset-confirm" className="block text-sm font-medium text-content-primary mb-1.5">
            Confirm new password
          </label>
          <div className="relative">
            <Lock className={`absolute left-3.5 top-1/2 -translate-y-1/2 w-4 h-4 pointer-events-none ${showError('confirm', confirmPassword) ? 'text-danger' : !passwordsMatch ? 'text-danger' : 'text-content-muted'}`} />
            <input
              id="reset-confirm"
              type={showConfirm ? 'text' : 'password'}
              value={confirmPassword}
              onChange={(e) => setConfirmPassword(e.target.value)}
              onBlur={() => handleBlur('confirm')}
              placeholder="Repeat your new password"
              autoComplete="new-password"
              className={`w-full pl-10 pr-11 py-2.5 bg-white border rounded-xl text-sm text-content-primary placeholder-content-muted focus:outline-none focus:ring-2 transition-all ${
                showError('confirm', confirmPassword)
                  ? 'border-danger focus:ring-danger/40 focus:border-danger bg-danger-50/30'
                  : !passwordsMatch
                    ? 'border-danger focus:ring-danger/40 focus:border-danger'
                    : 'border-border focus:ring-orange/40 focus:border-orange'
              }`}
            />
            <button
              type="button"
              onClick={() => setShowConfirm(!showConfirm)}
              className="absolute right-3 top-1/2 -translate-y-1/2 p-0.5 text-content-muted hover:text-content-secondary transition-colors"
              aria-label={showConfirm ? 'Hide password' : 'Show password'}
            >
              {showConfirm ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
            </button>
          </div>
          {showError('confirm', confirmPassword) && (
            <p className="text-xs text-danger mt-1.5">Please confirm your password</p>
          )}
          {!passwordsMatch && (
            <p className="text-xs text-danger mt-1.5">Passwords do not match</p>
          )}
        </div>

        {/* Password requirements */}
        <div className="bg-surface rounded-xl p-4 space-y-2">
          <p className="text-xs font-semibold text-content-primary mb-1">Password requirements:</p>
          {[
            { met: password.length >= 6, text: 'At least 6 characters' },
            { met: /[A-Z]/.test(password) && /[a-z]/.test(password), text: 'Upper & lowercase letters' },
            { met: /[0-9]/.test(password), text: 'At least one number' },
            { met: /[^A-Za-z0-9]/.test(password), text: 'At least one special character' },
          ].map((req) => (
            <div key={req.text} className="flex items-center gap-2">
              <CheckCircle2 className={`w-3.5 h-3.5 shrink-0 transition-colors ${req.met ? 'text-success' : 'text-border-dark'}`} />
              <span className={`text-xs transition-colors ${req.met ? 'text-content-primary' : 'text-content-muted'}`}>
                {req.text}
              </span>
            </div>
          ))}
        </div>

        {/* Submit */}
        <button
          type="submit"
          disabled={!passwordsMatch}
          className="w-full flex items-center justify-center gap-2 px-6 py-3 bg-orange hover:bg-orange-dark rounded-xl text-base font-semibold text-white transition-all shadow-md hover:shadow-lg hover:shadow-orange/20 active:scale-[0.98] disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:bg-orange disabled:active:scale-100 disabled:shadow-none"
        >
          Reset Password
          <ArrowRight className="w-4 h-4" />
        </button>
      </form>
    </div>
  );
};

export default ResetPasswordPage;
