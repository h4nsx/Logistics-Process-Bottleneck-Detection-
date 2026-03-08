import { Routes, Route } from 'react-router-dom';
import MainLayout from './layout/MainLayout';
import AuthLayout from './layout/AuthLayout';
import HomePage from '../pages/marketing/HomePage';
import CompanyPage from '../pages/marketing/CompanyPage';
import ContactPage from '../pages/marketing/ContactPage';
import LoginPage from '../pages/auth/LoginPage';
import RegisterPage from '../pages/auth/RegisterPage';
import ForgotPasswordPage from '../pages/auth/ForgotPasswordPage';
import DocumentationPage from '../pages/resources/Documentation';
import HelpCenterPage from '../pages/resources/HelpCenter';
import InteractiveDemoPage from '../pages/resources/InteractiveDemo';
import ApiReferencePage from '../pages/resources/ApiReference';
import ResetPasswordPage from '../pages/auth/ResetPasswordPage';
import PricingPage from '../pages/marketing/PricingPage';

/* Placeholder page for demonstration */
const PlaceholderPage = ({ title }: { title: string }) => (
  <div className="flex items-center justify-center min-h-[60vh]">
    <div className="text-center">
      <h1 className="text-3xl font-bold text-navy mb-2">{title}</h1>
      <p className="text-content-secondary">This page is under construction.</p>
    </div>
  </div>
);

const AppRouter = () => {
  return (
    <Routes>
      {/* Marketing pages use MainLayout (Navbar + Footer) */}
      <Route element={<MainLayout />}>
        <Route path="/" element={<HomePage />} />
        <Route path="/product" element={<PlaceholderPage title="Product" />} />
        <Route path="/product/:section" element={<PlaceholderPage title="Product" />} />
        <Route path="/solutions/:industry" element={<PlaceholderPage title="Solutions" />} />
        <Route path="/demo" element={<PlaceholderPage title="Demo" />} />
        <Route path="/demo/example" element={<InteractiveDemoPage />} />
        <Route path="/demo/:section" element={<PlaceholderPage title="Demo" />} />
        <Route path="/pricing" element={<PricingPage />} />
        <Route path="/resources/docs" element={<DocumentationPage />} />
        <Route path="/resources/help" element={<HelpCenterPage />} />
        <Route path="/resources/api" element={<ApiReferencePage />} />
        <Route path="/resources/:section" element={<PlaceholderPage title="Resources" />} />
        <Route path="/about-us/about" element={<CompanyPage />} />
        <Route path="/about-us/contact" element={<ContactPage />} />
        <Route path="/about-us/team" element={<PlaceholderPage title="Team" />} />
        <Route path="/about-us/:section" element={<PlaceholderPage title="About Us" />} />
      </Route>

      {/* Auth pages use AuthLayout (split-screen, no Navbar/Footer) */}
      <Route element={<AuthLayout />}>
        <Route path="/login" element={<LoginPage />} />
        <Route path="/register" element={<RegisterPage />} />
        <Route path="/forgot-password" element={<ForgotPasswordPage />} />
        <Route path="/reset-password" element={<ResetPasswordPage />} />
      </Route>
    </Routes>
  );
};

export default AppRouter;
