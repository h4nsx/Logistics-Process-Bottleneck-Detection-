import { useState } from 'react';
import {
  Clock,
  Send,
  ArrowRight,
  Github,
  MessageSquare,
  CheckCircle2,
} from 'lucide-react';

/* ──────────────────────────────────────────
   Section 1: Hero
   ────────────────────────────────────────── */

const HeroSection = () => (
  <section className="relative overflow-hidden bg-gradient-to-b from-navy-50 via-white to-white py-20 lg:py-28">
    <div className="absolute top-0 left-1/2 -translate-x-1/2 w-[800px] h-[400px] bg-gradient-to-b from-cyan/5 to-transparent rounded-full blur-3xl animate-pulse" style={{ animationDuration: '4s' }} />

    <div className="relative max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
      <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-cyan-50 border border-cyan-100 mb-6 opacity-0 animate-[fadeInUp_0.6s_ease-out_0.2s_forwards]">
        <MessageSquare className="w-3.5 h-3.5 text-cyan" />
        <span className="text-xs font-semibold text-cyan-dark tracking-wide uppercase">Contact Us</span>
      </div>

      <h1 className="text-4xl lg:text-5xl font-bold text-content-primary mb-6 leading-tight opacity-0 animate-[fadeInUp_0.7s_ease-out_0.4s_forwards]">
        Let's talk about your
        <br />
        <span className="text-orange">logistics challenges</span>
      </h1>

      <p className="text-lg text-content-secondary leading-relaxed max-w-2xl mx-auto opacity-0 animate-[fadeInUp_0.7s_ease-out_0.6s_forwards]">
        Whether you have questions about Vyn, need help with integration, or want to explore enterprise solutions — we're here to help.
      </p>
    </div>
  </section>
);


/* ──────────────────────────────────────────
   Section 2: Contact Form
   ────────────────────────────────────────── */

const ContactFormSection = () => {
  const [formData, setFormData] = useState({
    name: '',
    email: '',
    companyName: '',
    role: '',
    message: '',
  });
  const [submitted, setSubmitted] = useState(false);
  const [sending, setSending] = useState(false);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement | HTMLSelectElement>) => {
    setFormData((prev) => ({ ...prev, [e.target.name]: e.target.value }));
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setSending(true);
    // Simulate send
    setTimeout(() => {
      setSending(false);
      setSubmitted(true);
      console.log('Contact form submitted:', formData);
    }, 1500);
  };

  if (submitted) {
    return (
      <section className="py-16 lg:py-24 bg-surface">
        <div className="max-w-2xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
          <div className="bg-white rounded-3xl border border-border p-10 lg:p-16">
            <div className="w-16 h-16 rounded-full bg-success-50 flex items-center justify-center mx-auto mb-6 ring-4 ring-success/10">
              <CheckCircle2 className="w-8 h-8 text-success" />
            </div>
            <h2 className="text-2xl font-bold text-content-primary mb-2">
              Message sent!
            </h2>
            <p className="text-base text-content-secondary leading-relaxed mb-6">
              Thanks for reaching out. We'll get back to you within 24 hours.
            </p>
            <button
              onClick={() => {
                setSubmitted(false);
                setFormData({ name: '', email: '', companyName: '', role: '', message: '' });
              }}
              className="text-sm font-medium text-orange hover:text-orange-dark transition-colors"
            >
              Send another message
            </button>
          </div>
        </div>
      </section>
    );
  }

  return (
    <section className="py-16 lg:py-24 bg-surface">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="grid grid-cols-1 lg:grid-cols-5 gap-12 lg:gap-16">

          {/* Left: Info */}
          <div className="lg:col-span-2">
            <h2 className="text-3xl font-bold text-content-primary mb-4">
              Send us a message
            </h2>
            <p className="text-base text-content-secondary leading-relaxed mb-8">
              Fill out the form and our team will respond within 24 hours. For urgent matters, email us directly.
            </p>

            <div className="space-y-4">
              <div className="flex items-start gap-3">
                <div className="w-9 h-9 rounded-lg bg-orange-50 flex items-center justify-center shrink-0">
                  <Clock className="w-4 h-4 text-orange" />
                </div>
                <div>
                  <p className="text-sm font-semibold text-content-primary">Response time</p>
                  <p className="text-xs text-content-muted">Usually within 24 hours</p>
                </div>
              </div>
              <div className="flex items-start gap-3">
                <div className="w-9 h-9 rounded-lg bg-cyan-50 flex items-center justify-center shrink-0">
                  <MessageSquare className="w-4 h-4 text-cyan" />
                </div>
                <div>
                  <p className="text-sm font-semibold text-content-primary">Enterprise inquiries</p>
                  <p className="text-xs text-content-muted">Custom plans and dedicated support</p>
                </div>
              </div>
              <div className="flex items-start gap-3">
                <div className="w-9 h-9 rounded-lg bg-success-50 flex items-center justify-center shrink-0">
                  <Github className="w-4 h-4 text-success" />
                </div>
                <div>
                  <p className="text-sm font-semibold text-content-primary">Open source</p>
                  <p className="text-xs text-content-muted">Report bugs on GitHub for fastest response</p>
                </div>
              </div>
            </div>
          </div>

          {/* Right: Form */}
          <div className="lg:col-span-3">
            <form onSubmit={handleSubmit} className="bg-white rounded-2xl border border-border p-6 lg:p-8 space-y-5">
              {/* Row 1: Full Name + Work Email */}
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-5">
                <div>
                  <label htmlFor="contact-name" className="block text-sm font-medium text-content-primary mb-1.5">
                    Full Name <span className="text-danger">*</span>
                  </label>
                  <input
                    id="contact-name"
                    name="name"
                    type="text"
                    value={formData.name}
                    onChange={handleChange}
                    placeholder="Jane Smith"
                    required
                    className="w-full px-4 py-2.5 bg-white border border-border rounded-xl text-sm text-content-primary placeholder-content-muted focus:outline-none focus:ring-2 focus:ring-orange/40 focus:border-orange transition-all"
                  />
                </div>
                <div>
                  <label htmlFor="contact-email" className="block text-sm font-medium text-content-primary mb-1.5">
                    Work Email <span className="text-danger">*</span>
                  </label>
                  <input
                    id="contact-email"
                    name="email"
                    type="email"
                    value={formData.email}
                    onChange={handleChange}
                    placeholder="jane@company.com"
                    required
                    className="w-full px-4 py-2.5 bg-white border border-border rounded-xl text-sm text-content-primary placeholder-content-muted focus:outline-none focus:ring-2 focus:ring-orange/40 focus:border-orange transition-all"
                  />
                </div>
              </div>

              {/* Row 2: Company Name + Role */}
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-5">
                <div>
                  <label htmlFor="contact-company" className="block text-sm font-medium text-content-primary mb-1.5">
                    Company Name <span className="text-danger">*</span>
                  </label>
                  <input
                    id="contact-company"
                    name="companyName"
                    type="text"
                    value={formData.companyName}
                    onChange={handleChange}
                    placeholder="Acme Logistics Co."
                    required
                    className="w-full px-4 py-2.5 bg-white border border-border rounded-xl text-sm text-content-primary placeholder-content-muted focus:outline-none focus:ring-2 focus:ring-orange/40 focus:border-orange transition-all"
                  />
                </div>
                <div>
                  <label htmlFor="contact-role" className="block text-sm font-medium text-content-primary mb-1.5">
                    Your Role <span className="text-danger">*</span>
                  </label>
                  <select
                    id="contact-role"
                    name="role"
                    value={formData.role}
                    onChange={handleChange}
                    required
                    className="w-full px-4 py-2.5 bg-white border border-border rounded-xl text-sm text-content-primary focus:outline-none focus:ring-2 focus:ring-orange/40 focus:border-orange transition-all appearance-none"
                  >
                    <option value="" disabled>Select your role</option>
                    <option value="operations">Operations Manager</option>
                    <option value="logistics">Logistics Manager</option>
                    <option value="analyst">Data Analyst</option>
                    <option value="supply-chain">Supply Chain Lead</option>
                    <option value="other">Other</option>
                  </select>
                </div>
              </div>

              {/* Message */}
              <div>
                <label htmlFor="contact-message" className="block text-sm font-medium text-content-primary mb-1.5">
                  Message
                </label>
                <textarea
                  id="contact-message"
                  name="message"
                  value={formData.message}
                  onChange={handleChange}
                  placeholder="Tell us how we can help..."
                  required
                  rows={5}
                  className="w-full px-4 py-2.5 bg-white border border-border rounded-xl text-sm text-content-primary placeholder-content-muted focus:outline-none focus:ring-2 focus:ring-orange/40 focus:border-orange transition-all resize-none"
                />
              </div>

              {/* Submit */}
              <button
                type="submit"
                disabled={sending}
                className="w-full flex items-center justify-center gap-2 px-6 py-3 bg-orange hover:bg-orange-dark rounded-xl text-base font-semibold text-white transition-all shadow-md hover:shadow-lg hover:shadow-orange/20 active:scale-[0.98] disabled:opacity-70 disabled:cursor-wait"
              >
                {sending ? (
                  <>
                    <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                    Sending...
                  </>
                ) : (
                  <>
                    Send Message
                    <Send className="w-4 h-4" />
                  </>
                )}
              </button>
            </form>
          </div>

        </div>
      </div>
    </section>
  );
};

/* ──────────────────────────────────────────
   FAQ Section
   ────────────────────────────────────────── */

const faqs = [
  {
    q: 'How quickly can I get started with Vyn?',
    a: 'You can sign up and upload your first CSV dataset in under 5 minutes. No data science background required — just prepare your execution timestamps in the required 5-column format and upload.',
  },
  {
    q: 'Do I need any technical skills to use Vyn?',
    a: 'No. If you can export a spreadsheet to CSV, you can use Vyn. The platform automatically computes baselines and flags bottlenecks. No SQL, no code, no data pipelines needed.',
  },
  {
    q: 'What data formats does Vyn support?',
    a: 'Vyn currently accepts CSV files only. Your file must contain five columns: process_id, step_code, start_time, end_time, and location. API integrations are planned for a future release.',
  },
  {
    q: 'Does Vyn offer enterprise support?',
    a: 'Yes — our Enterprise plan includes dedicated support, custom risk thresholds, and unlimited CSV uploads. Use the contact form above and select "Operations Manager" or "Logistics Manager" to start an enterprise conversation.',
  },
];

const FAQSection = () => (
  <section className="py-16 lg:py-24 bg-white">
    <div className="max-w-3xl mx-auto px-4 sm:px-6 lg:px-8">
      <div className="text-center mb-12">
        <h2 className="text-3xl font-bold text-content-primary mb-4">
          Frequently asked questions
        </h2>
        <p className="text-lg text-content-secondary">
          Can't find what you need? <a href="mailto:hello@vyn.dev" className="text-orange hover:text-orange-dark font-medium transition-colors">Email us</a>
        </p>
      </div>

      <div className="space-y-4">
        {faqs.map((faq) => (
          <details
            key={faq.q}
            className="group bg-white rounded-2xl border border-border overflow-hidden hover:border-orange/20 transition-colors"
          >
            <summary className="flex items-center justify-between gap-4 px-6 py-4 cursor-pointer text-sm font-semibold text-content-primary select-none list-none">
              {faq.q}
              <ArrowRight className="w-4 h-4 text-content-muted shrink-0 transition-transform duration-200 group-open:rotate-90" />
            </summary>
            <div className="px-6 pb-4 text-sm text-content-secondary leading-relaxed">
              {faq.a}
            </div>
          </details>
        ))}
      </div>
    </div>
  </section>
);

/* ──────────────────────────────────────────
   Contact Page
   ────────────────────────────────────────── */

const ContactPage = () => {
  return (
    <>
      <HeroSection />
      <ContactFormSection />
      <FAQSection />
    </>
  );
};

export default ContactPage;
