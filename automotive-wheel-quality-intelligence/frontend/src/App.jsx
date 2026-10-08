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
          <p style={{ fontSize: '0.85rem', color: '#94a3b8', marginTop: '0.2rem' }}>
            SINGULARITY 2026 — Track 3 Quality Intelligence Cockpit
          </p>
        </div>
        <span
          style={{
            backgroundColor: '#0369a1',
            color: '#e0f2fe',
            fontSize: '0.75rem',
            fontWeight: '700',
            padding: '0.35rem 0.85rem',
            borderRadius: '9999px',
            textTransform: 'uppercase',
            letterSpacing: '0.05em',
            border: '1px solid #0284c7',
          }}
        >
          LIVE PROTOTYPE
        </span>
      </header>

      <main>
        <Dashboard />
      </main>
    </div>
  );
}
