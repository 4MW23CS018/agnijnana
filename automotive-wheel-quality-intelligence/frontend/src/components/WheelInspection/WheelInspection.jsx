import React, { useState, useRef, useEffect } from 'react';
import { uploadWheelImage, inspectWheel } from '../../services/api';

// ── Severity colour for confidence badges ──────────────────────────────────
function confColor(v) {
  if (v === undefined || v === null) return '#64748b';
  if (v >= 0.75) return '#ef4444';
  if (v >= 0.45) return '#f59e0b';
  return '#22c55e';
}

function pct(v) {
  return v !== undefined && v !== null ? `${(v * 100).toFixed(1)}%` : 'N/A';
}

// ── Bounding-box overlay on the image canvas ───────────────────────────────
function BBoxOverlay({ detections, imgNaturalSize }) {
  if (!detections?.length || !imgNaturalSize) return null;
  const { naturalW, naturalH, displayW, displayH } = imgNaturalSize;
  const scaleX = displayW / naturalW;
  const scaleY = displayH / naturalH;

  return (
    <svg
      viewBox={`0 0 ${displayW} ${displayH}`}
      preserveAspectRatio="none"
      style={{
        position: 'absolute',
        inset: 0,
        width: '100%',
        height: '100%',
        pointerEvents: 'none',
      }}
    >
      {detections.map((ld, i) => {
        const [x1, y1, x2, y2] = ld.localization.bbox;
        const rx = x1 * scaleX;
        const ry = y1 * scaleY;
        const rw = (x2 - x1) * scaleX;
        const rh = (y2 - y1) * scaleY;
        const agreed = ld.classification_agreement;
        const strokeCol = agreed ? '#22c55e' : '#f59e0b';
        const label = ld.classification.cnn_defect_type.toUpperCase();
        const conf = pct(ld.classification.cnn_confidence);
        return (
          <g key={i}>
            <rect
              x={rx} y={ry} width={rw} height={rh}
              fill="none"
              stroke={strokeCol}
              strokeWidth="2"
              strokeDasharray={agreed ? '0' : '5,3'}
            />
            {/* Label background */}
            <rect
              x={rx} y={Math.max(0, ry - 22)}
              width={Math.min(rw, 160)} height={20}
              fill={agreed ? '#16a34a' : '#d97706'}
              rx={3}
            />
            <text
              x={rx + 4} y={Math.max(0, ry - 7)}
              fill="#ffffff"
              fontSize="11"
              fontWeight="bold"
              fontFamily="monospace"
            >
              {label} {conf}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

// ── Main component ─────────────────────────────────────────────────────────
export default function WheelInspection() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [errorMsg, setErrorMsg] = useState(null);
  const [imgNaturalSize, setImgNaturalSize] = useState(null);
  const imgRef = useRef(null);

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
      setResult(null);
      setErrorMsg(null);
      setImgNaturalSize(null);
    }
  };

  const handleUploadAndInspect = async () => {
    if (!selectedFile) { setErrorMsg('Select a wheel image first.'); return; }
    setLoading(true);
    setErrorMsg(null);
    try {
      const uploadRes = await uploadWheelImage(selectedFile);
      const wheelId = `WHEEL-${Math.floor(1000 + Math.random() * 9000)}`;
      const data = await inspectWheel(wheelId, uploadRes.image_path, 'LOT-2026-A', 'DIE-CAST-01');
      setResult(data);
    } catch (err) {
      setErrorMsg(err.message || 'Inspection failed.');
    } finally {
      setLoading(false);
    }
  };

  const handleImgLoad = () => {
    const el = imgRef.current;
    if (el) {
      setImgNaturalSize({
        naturalW: el.naturalWidth,
        naturalH: el.naturalHeight,
        displayW: el.width,
        displayH: el.height,
      });
    }
  };

  // Re-measure on window resize
  useEffect(() => {
    const onResize = () => {
      const el = imgRef.current;
      if (el) {
        setImgNaturalSize({
          naturalW: el.naturalWidth,
          naturalH: el.naturalHeight,
          displayW: el.width,
          displayH: el.height,
        });
      }
    };
    window.addEventListener('resize', onResize);
    return () => window.removeEventListener('resize', onResize);
  }, []);

  // ── Parse result ───────────────────────────────────────────────────────────
  const hybrid = result?.hybrid;
  const rim = result?.rim;                      // CNN-only fallback
  const detections = hybrid?.localized_defects ?? [];
  const locStatus = hybrid?.localization_status;

  // Primary defect/confidence: first localized CNN result, or full-image CNN
  const fullCnn = hybrid?.full_image_cnn ?? {};
  const primaryDefect = detections.length > 0
    ? detections[0].classification.cnn_defect_type
    : (fullCnn.defect_type ?? rim?.defect_type ?? 'N/A');
  const primaryConf = detections.length > 0
    ? detections[0].classification.cnn_confidence
    : (fullCnn.confidence ?? rim?.confidence);

  const yoloMs = hybrid?.yolo_inference_ms;
  const cnnMs = hybrid?.cnn_total_inference_ms;
  const totalMs = hybrid?.total_hybrid_ms;

  return (
    <div className="card-placeholder col-12" style={{ textAlign: 'left', padding: '1.25rem' }}>

      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
        <h3 style={{ margin: 0, color: '#f8fafc', fontSize: '1.15rem' }}>
          Rim Defect Visual Inspection
        </h3>
        <div style={{ display: 'flex', gap: '0.5rem' }}>
          <span style={{ fontSize: '0.7rem', fontWeight: '700', backgroundColor: '#0369a1', color: '#e0f2fe', padding: '0.2rem 0.55rem', borderRadius: '4px', textTransform: 'uppercase' }}>
            ConvNeXt-Tiny
          </span>
          <span style={{ fontSize: '0.7rem', fontWeight: '700', backgroundColor: '#7c3aed', color: '#ede9fe', padding: '0.2rem 0.55rem', borderRadius: '4px', textTransform: 'uppercase' }}>
            YOLO26s-seg
          </span>
        </div>
      </div>

      {/* Upload controls */}
      <div style={{ display: 'flex', gap: '1rem', alignItems: 'center', marginBottom: '1.25rem', flexWrap: 'wrap' }}>
        <input
          type="file"
          accept="image/jpeg,image/png,image/webp"
          onChange={handleFileChange}
          style={{ padding: '0.45rem 0.75rem', backgroundColor: '#0f172a', border: '1px solid #334155', borderRadius: '6px', color: '#f8fafc', fontSize: '0.875rem' }}
        />
        <button
          onClick={handleUploadAndInspect}
          disabled={loading || !selectedFile}
          style={{ padding: '0.55rem 1.25rem', backgroundColor: loading ? '#64748b' : '#7c3aed', color: '#ffffff', border: 'none', borderRadius: '6px', fontWeight: '700', fontSize: '0.875rem', cursor: loading || !selectedFile ? 'not-allowed' : 'pointer', transition: 'background-color 0.2s' }}
        >
          {loading ? '⟳ Running Hybrid Inference…' : '⬆ Upload & Inspect Wheel'}
        </button>
      </div>

      {/* Error */}
      {errorMsg && (
        <div style={{ padding: '0.75rem 1rem', backgroundColor: '#881337', color: '#fecdd3', borderRadius: '6px', marginBottom: '1rem', border: '1px solid #9f1239', fontSize: '0.875rem' }}>
          <strong>Error:</strong> {errorMsg}
        </div>
      )}

      {/* Main grid: image | results */}
      <div style={{ display: 'grid', gridTemplateColumns: 'minmax(260px, 1fr) 2fr', gap: '1.25rem', marginTop: '0.5rem' }}>

        {/* Image preview + YOLO bbox overlay */}
        <div style={{ backgroundColor: '#0f172a', border: '1px solid #334155', borderRadius: '8px', padding: '0.75rem', textAlign: 'center', minHeight: '220px', display: 'flex', flexDirection: 'column', justifyContent: 'center', alignItems: 'center' }}>
          {previewUrl ? (
            <div style={{ position: 'relative', display: 'inline-block' }}>
              <img
                ref={imgRef}
                src={previewUrl}
                alt="Uploaded Wheel"
                onLoad={handleImgLoad}
                style={{ maxWidth: '100%', maxHeight: '260px', objectFit: 'contain', borderRadius: '4px', display: 'block' }}
              />
              {result && detections.length > 0 && (
                <BBoxOverlay
                  detections={detections}
                  imgNaturalSize={imgNaturalSize}
                />
              )}
            </div>
          ) : (
            <div style={{ padding: '2rem 1rem', color: '#64748b', fontSize: '0.875rem' }}>
              <div style={{ fontSize: '1.5rem', marginBottom: '0.5rem' }}>📷</div>
              Select a wheel image to begin inspection
            </div>
          )}
          {result && detections.length > 0 && (
            <div style={{ marginTop: '0.5rem', fontSize: '0.7rem', color: '#a78bfa' }}>
              {detections.length} YOLO defect region{detections.length > 1 ? 's' : ''} detected
            </div>
          )}
        </div>

        {/* Results panel */}
        <div style={{ backgroundColor: '#0f172a', border: '1px solid #334155', borderRadius: '8px', padding: '1.25rem', display: 'flex', flexDirection: 'column', gap: '1rem' }}>

          {result ? (
            <>
              {/* Primary Classification */}
              <div>
                <div style={{ fontSize: '0.7rem', fontWeight: '700', color: '#94a3b8', textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: '0.6rem' }}>
                  Classification Result
                </div>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(130px, 1fr))', gap: '0.75rem' }}>
                  <Stat label="Defect" value={primaryDefect?.toUpperCase()} large accent="#f43f5e" />
                  <Stat label="CNN Confidence" value={pct(primaryConf)} large accent="#38bdf8" />
                  <Stat label="Latency (YOLO)" value={yoloMs !== undefined ? `${yoloMs} ms` : '—'} />
                  <Stat label="Latency (CNN)" value={cnnMs !== undefined ? `${cnnMs} ms` : '—'} />
                  <Stat label="Total Hybrid" value={totalMs !== undefined ? `${totalMs} ms` : '—'} />
                  <Stat label="Device" value={(fullCnn.device ?? rim?.device ?? 'cpu').toUpperCase()} />
                </div>
              </div>

              {/* Localization section */}
              <div style={{ borderTop: '1px solid #1e293b', paddingTop: '0.75rem' }}>
                <div style={{ fontSize: '0.7rem', fontWeight: '700', color: '#94a3b8', textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: '0.6rem' }}>
                  YOLO Localization
                </div>

                {locStatus === 'no_yolo_detection' ? (
                  <StatusPill label="No YOLO Detection" color="#334155" textColor="#94a3b8"
                    note={`Threshold: ${(hybrid?.yolo_conf_threshold * 100).toFixed(0)}% — full-image CNN used as fallback`}
                  />
                ) : detections.length > 0 ? (
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                    {detections.map((ld, i) => (
                      <DetectionCard key={i} ld={ld} index={i} />
                    ))}
                  </div>
                ) : (
                  <StatusPill label="Pending" color="#334155" textColor="#94a3b8" />
                )}
              </div>

              {/* Full-image CNN baseline */}
              {fullCnn.defect_type && (
                <div style={{ borderTop: '1px solid #1e293b', paddingTop: '0.75rem' }}>
                  <div style={{ fontSize: '0.7rem', fontWeight: '700', color: '#94a3b8', textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: '0.4rem' }}>
                    Full-Image CNN Baseline
                  </div>
                  <div style={{ fontSize: '0.82rem', color: '#cbd5e1' }}>
                    <span style={{ color: '#f8fafc', fontWeight: '700' }}>{fullCnn.defect_type?.toUpperCase()}</span>
                    {' '}—{' '}{pct(fullCnn.confidence)} confidence
                    <span style={{ marginLeft: '0.75rem', color: '#64748b', fontSize: '0.75rem' }}>
                      ({fullCnn.inference_ms} ms, {fullCnn.model})
                    </span>
                  </div>
                </div>
              )}
            </>
          ) : (
            <div style={{ color: '#64748b', fontStyle: 'italic', padding: '1.5rem 0', fontSize: '0.875rem' }}>
              No active inspection result. Upload a wheel image to inspect.
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

// ── Sub-components ─────────────────────────────────────────────────────────

function Stat({ label, value, large, accent }) {
  return (
    <div>
      <small style={{ color: '#94a3b8', fontSize: '0.72rem' }}>{label}</small>
      <div style={{
        fontSize: large ? '1.35rem' : '0.92rem',
        fontWeight: large ? '800' : '600',
        color: accent ?? '#e2e8f0',
        marginTop: '0.15rem',
        textTransform: large ? 'uppercase' : 'none',
      }}>
        {value ?? '—'}
      </div>
    </div>
  );
}

function StatusPill({ label, color, textColor, note }) {
  return (
    <div>
      <span style={{ fontSize: '0.72rem', fontWeight: '700', padding: '0.2rem 0.6rem', borderRadius: '4px', backgroundColor: color, color: textColor, letterSpacing: '0.05em' }}>
        {label}
      </span>
      {note && <span style={{ marginLeft: '0.6rem', fontSize: '0.72rem', color: '#64748b' }}>{note}</span>}
    </div>
  );
}

function DetectionCard({ ld, index }) {
  const loc = ld.localization;
  const cls = ld.classification;
  const agreed = ld.classification_agreement;
  const [x1, y1, x2, y2] = loc.bbox;

  return (
    <div style={{
      padding: '0.6rem 0.85rem',
      backgroundColor: '#111827',
      borderRadius: '6px',
      border: `1px solid ${agreed ? '#166534' : '#92400e'}`,
      display: 'grid',
      gridTemplateColumns: 'repeat(auto-fit, minmax(120px, 1fr))',
      gap: '0.5rem 1rem',
      fontSize: '0.8rem',
    }}>
      <div>
        <div style={{ color: '#64748b', fontSize: '0.68rem', textTransform: 'uppercase', marginBottom: '0.15rem' }}>YOLO Region #{index + 1}</div>
        <div style={{ color: '#e2e8f0', fontWeight: '700' }}>{loc.yolo_defect_type?.toUpperCase()}</div>
        <div style={{ color: '#94a3b8' }}>conf: {pct(loc.yolo_confidence)}</div>
        <div style={{ color: '#475569', fontSize: '0.68rem', fontFamily: 'monospace', marginTop: '0.15rem' }}>
          [{x1.toFixed(0)},{y1.toFixed(0)} → {x2.toFixed(0)},{y2.toFixed(0)}]
        </div>
      </div>
      <div>
        <div style={{ color: '#64748b', fontSize: '0.68rem', textTransform: 'uppercase', marginBottom: '0.15rem' }}>CNN Classification</div>
        <div style={{ color: '#f8fafc', fontWeight: '700' }}>{cls.cnn_defect_type?.toUpperCase()}</div>
        <div style={{ color: '#38bdf8' }}>conf: {pct(cls.cnn_confidence)}</div>
        <div style={{ color: '#475569', fontSize: '0.68rem', marginTop: '0.15rem' }}>
          {cls.inference_ms} ms · {cls.device?.toUpperCase()}
        </div>
      </div>
      <div style={{ display: 'flex', alignItems: 'center' }}>
        <span style={{
          fontSize: '0.68rem', fontWeight: '700',
          padding: '0.2rem 0.5rem', borderRadius: '4px',
          backgroundColor: agreed ? '#14532d' : '#7c2d12',
          color: agreed ? '#86efac' : '#fed7aa',
        }}>
          {agreed ? '✓ AGREE' : '⚠ DISAGREE'}
        </span>
        {!agreed && (
          <span style={{ marginLeft: '0.4rem', fontSize: '0.66rem', color: '#78716c' }}>
            YOLO:{loc.yolo_defect_type} ≠ CNN:{cls.cnn_defect_type}
          </span>
        )}
      </div>
    </div>
  );
}
