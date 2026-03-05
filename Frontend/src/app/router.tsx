import { Routes, Route } from 'react-router-dom';
import MainLayout from './layout/MainLayout';
import HomePage from '../pages/marketing/HomePage';

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
      {/* Marketing pages use MainLayout */}
      <Route element={<MainLayout />}>
        <Route path="/" element={<HomePage />} />
        <Route path="/product" element={<PlaceholderPage title="Product" />} />
        <Route path="/product/:section" element={<PlaceholderPage title="Product" />} />
        <Route path="/solutions/:industry" element={<PlaceholderPage title="Solutions" />} />
        <Route path="/demo" element={<PlaceholderPage title="Demo" />} />
        <Route path="/demo/:section" element={<PlaceholderPage title="Demo" />} />
        <Route path="/pricing" element={<PlaceholderPage title="Pricing" />} />
        <Route path="/resources/:section" element={<PlaceholderPage title="Resources" />} />
        <Route path="/about-us/:section" element={<PlaceholderPage title="About Us" />} />
        <Route path="/login" element={<PlaceholderPage title="Sign In" />} />
        <Route path="/register" element={<PlaceholderPage title="Create Account" />} />
      </Route>
    </Routes>
  );
};

export default AppRouter;
