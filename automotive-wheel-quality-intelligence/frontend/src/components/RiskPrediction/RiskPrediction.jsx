import React from 'react';

export default function RiskPrediction() {
  return (
    <div className="card-placeholder col-6" style={{ textTransform: 'none' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.75rem' }}>
        <h3 style={{ margin: 0, color: '#f8fafc', fontSize: '1rem' }}>Future Risk Forecast</h3>
        <span
          style={{
            fontSize: '0.7rem',
            fontWeight: '700',
            padding: '0.15rem 0.5rem',
            borderRadius: '4px',
            backgroundColor: '#334155',
            color: '#94a3b8',
            letterSpacing: '0.05em',
          }}
        >
          PENDING
        </span>
      </div>
      <div
        style={{
          flex: 1,
          backgroundColor: '#0f172a',
          border: '1px solid #334155',
          borderRadius: '6px',
          padding: '1rem',
          display: 'flex',
          flexDirection: 'column',
          justifyContent: 'center',
          color: '#94a3b8',
          fontSize: '0.8125rem',
          lineHeight: '1.4',
        }}
      >
        Will estimate machines and future batches at risk from historical quality trends.
      </div>
    </div>
  );
}
