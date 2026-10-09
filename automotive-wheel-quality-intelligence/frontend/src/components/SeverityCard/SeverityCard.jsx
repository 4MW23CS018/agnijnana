
import React from 'react';

export default function SeverityCard({ inspectionResult }) {
  const result = inspectionResult;

  const severity = result?.severity ?? null;
  const score = result?.severity_score ?? null;
  const rationale = result?.severity_rationale ?? null;

  const defects = result?.hybrid?.localized_defects ?? [];
  const hasResult = result != null;

  const statusColor =
    severity === 'Critical' ? '#ef4444'
      : severity === 'Low' ? '#22c55e'
        : severity === 'Uncertain' ? '#f59e0b'
          : '#94a3b8';

  return (
    <div className="card-placeholder col-4" style={{ textTransform: 'none' }}>
      <div style={{
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        marginBottom: '0.75rem',
      }}>
        <h3 style={{ margin: 0, color: '#f8fafc', fontSize: '1rem' }}>
          Severity Analysis
        </h3>

        <span style={{
          fontSize: '0.7rem',
          fontWeight: '700',
          padding: '0.15rem 0.5rem',
          borderRadius: '4px',
          backgroundColor: hasResult ? `${statusColor}25` : '#334155',
          color: hasResult ? statusColor : '#94a3b8',
          letterSpacing: '0.05em',
        }}>
          {hasResult ? severity ?? 'AVAILABLE' : 'PENDING'}
        </span>
      </div>

      <div style={{
        backgroundColor: '#0f172a',
        border: '1px solid #334155',
        borderRadius: '6px',
        padding: '1rem',
        color: '#cbd5e1',
        fontSize: '0.8125rem',
        lineHeight: '1.5',
      }}>
        {!hasResult ? (
          'Upload and inspect a rim image to see its severity assessment.'
        ) : (
          <>
            <div style={{
              display: 'flex',
              justifyContent: 'space-between',
              gap: '1rem',
              marginBottom: '0.75rem',
            }}>
              <div>
                <div style={{ color: '#94a3b8', fontSize: '0.72rem' }}>
                  Severity level
                </div>
                <strong style={{ color: statusColor }}>
                  {severity ?? 'Not available'}
                </strong>
              </div>

              <div>
                <div style={{ color: '#94a3b8', fontSize: '0.72rem' }}>
                  Severity score
                </div>
                <strong>
                  {score != null ? `${score}/100` : '—'}
                </strong>
              </div>

              <div>
                <div style={{ color: '#94a3b8', fontSize: '0.72rem' }}>
                  Defects detected
                </div>
                <strong>{defects.length}</strong>
              </div>
            </div>

            {rationale && (
              <div>
                <strong style={{ color: '#38bdf8' }}>Rationale: </strong>
                {rationale}
              </div>
            )}

            {defects.length > 0 && (
              <div style={{ marginTop: '0.75rem' }}>
                <strong>Regional severity</strong>

                {defects.map((defect, index) => (
                  <div
                    key={index}
                    style={{
                      display: 'flex',
                      justifyContent: 'space-between',
                      gap: '0.5rem',
                      marginTop: '0.4rem',
                      padding: '0.5rem',
                      backgroundColor: '#111827',
                      borderRadius: '4px',
                    }}
                  >
                    <span>
                      {defect.localization?.yolo_defect_type ?? 'Defect'} #{index + 1}
                    </span>

                    <strong style={{
                      color: defect.severity_level === 'Critical'
                        ? '#ef4444'
                        : defect.severity_level === 'Low'
                          ? '#22c55e'
                          : '#f59e0b',
                    }}>
                      {defect.severity_level ?? 'Unknown'}
                      {defect.severity_score != null
                        ? ` (${defect.severity_score}/100)`
                        : ''}
                    </strong>
                  </div>
                ))}
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}
