import React from 'react';

const App = () => {
    return (
        <div className="flex min-h-screen flex-col items-center justify-center bg-slate-900 text-white font-sans">
            <div className="rounded-xl bg-slate-800 p-8 shadow-2xl border border-slate-700 text-center max-w-2xl">
                <h1 className="text-4xl font-extrabold text-blue-400 mb-4 tracking-tight">
                    Logistics Process Bottleneck Detection
                </h1>
                <p className="text-slate-300 text-lg mb-8">
                    The frontend project has been successfully initialized. You are ready to start building features!
                </p>
                <div className="flex justify-center gap-4">
                    <div className="bg-slate-700/50 px-4 py-2 rounded-lg text-sm text-blue-300 border border-slate-600">
                        React 19 + Vite
                    </div>
                    <div className="bg-slate-700/50 px-4 py-2 rounded-lg text-sm text-cyan-300 border border-slate-600">
                        Tailwind CSS
                    </div>
                    <div className="bg-slate-700/50 px-4 py-2 rounded-lg text-sm text-emerald-300 border border-slate-600">
                        TanStack Query
                    </div>
                </div>
            </div>
        </div>
    );
};

export default App;
