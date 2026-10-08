import React from 'react';

export default function SystemStatus() {
  const statusItems = [
    { label: 'Vision Classification', status: 'READY', type: 'ready' },
    { label: 'Image Upload', status: 'READY', type: 'ready' },
    { label: 'Backend API', status: 'CONNECTED', type: 'connected' },
    { label: 'Localization (YOLO)', status: 'PENDING', type: 'pending' },
    { label: 'Severity Analysis', status: 'PENDING', type: 'pending' },
    { label: 'Root Cause (RCA)', status: 'PENDING', type: 'pending' },
    { label: 'Risk Forecast', status: 'PENDING', type: 'pending' },
    { label: 'Corrective Action', status: 'PENDING', type: 'pending' },
    { label: 'Batch Alerts', status: 'PENDING', type: 'pending' },
  ];

  return (
    <div className="card-placeholder col-12" style={{ textTransform: 'none' }}>
      <h3 style={{ margin: '0 0 1rem 0', color: '#f8fafc', fontSize: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
        <span style={{ width: '8px', height: '8px', borderRadius: '50%', backgroundColor: '#10b981', display: 'inline-block' }}></span>
        System Operational Status
      </h3>
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
          gap: '0.75rem',
        }}
      >
        {statusItems.map((item, index) => {
          let badgeBg = '#334155';
          let badgeColor = '#94a3b8';

          if (item.type === 'ready') {
            badgeBg = '#065f46';
            badgeColor = '#6ee7b7';
          } else if (item.type === 'connected') {
            badgeBg = '#0369a1';
            badgeColor = '#7dd3fc';
          }

          return (
            <div
              key={index}
              style={{
                backgroundColor: '#0f172a',
                border: '1px solid #334155',
                borderRadius: '6px',
                padding: '0.6rem 0.85rem',
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
              }}
            >
              <span style={{ fontSize: '0.8125rem', color: '#cbd5e1', fontWeight: '500' }}>
                {item.label}
              </span>
              <span
                style={{
                  fontSize: '0.7rem',
                  fontWeight: '700',
                  padding: '0.15rem 0.5rem',
                  borderRadius: '4px',
                  backgroundColor: badgeBg,
                  color: badgeColor,
                  letterSpacing: '0.05em',
                }}
              >
                {item.status}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
