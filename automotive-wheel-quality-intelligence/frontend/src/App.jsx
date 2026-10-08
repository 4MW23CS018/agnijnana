import React, { useState } from 'react';
import Dashboard from './pages/Dashboard/Dashboard';
import './App.css';

export default function App() {
  const [activeTab, setActiveTab] = useState('dashboard');

  return (
    <div className="app-container">
      <header className="app-header">
        <div>
          <h1 className="app-title">Aluminium Wheel Quality Intelligence System</h1>
          <p style={{ fontSize: '0.875rem', color: '#94a3b8' }}>
            SINGULARITY 2026 — Track 3 Quality Intelligence Cockpit (Aluminium Alloy Wheel Inspection)
          </p>
        </div>
        <span className="phase-badge">Skeleton / Foundation Phase</span>
      </header>

      <main>
        <Dashboard />
      </main>
    </div>
  );
}
