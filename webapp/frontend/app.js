/* PCB Defect Detection — Premium React App */
const { useState, useRef, useEffect, useCallback } = React;
const e = React.createElement;

const CLASS_COLORS = {
  missing_hole: '#ef4444',
  mouse_bite: '#f97316',
  open_circuit: '#eab308',
  short: '#a855f7',
  spur: '#06b6d4',
  spurious_copper: '#22c55e',
};

const CLASS_LABELS = Object.keys(CLASS_COLORS).map(k => ({
  key: k,
  label: k.replace(/_/g, ' '),
  color: CLASS_COLORS[k],
}));

function classColor(name) {
  const k = name.toLowerCase().replace(/ /g, '_');
  return CLASS_COLORS[k] || '#94a3b8';
}

// ---------- Small UI helpers ---------- //

function Badge({ count, color }) {
  return e('span', {
    className: 'defect-count-badge',
    style: { background: color + '22', color, border: `1px solid ${color}44` }
  }, count);
}

function StatCard({ value, label }) {
  return e('div', { className: 'stat-card' },
    e('div', { className: 'stat-value' }, value),
    e('div', { className: 'stat-label' }, label)
  );
}

function ClassLegend() {
  return e('div', { className: 'class-legend' },
    e('div', { className: 'class-legend-title' }, 'Defect Classes'),
    e('div', { className: 'class-chips' },
      ...CLASS_LABELS.map(c =>
        e('span', {
          key: c.key,
          className: 'class-chip',
          style: { color: c.color, borderColor: c.color + '44', background: c.color + '11' }
        },
          e('span', { style: { width: 8, height: 8, borderRadius: '50%', background: c.color, display: 'inline-block' } }),
          c.label
        )
      )
    )
  );
}

function NoModelBanner() {
  return e('div', { className: 'alert-banner' },
    e('span', { style: { fontSize: 20 } }, '⚠️'),
    e('div', null,
      e('strong', null, 'No PCB model detected — using generic model'),
      'Upload an image to test detection. For real PCB defect results, train a model on Kaggle and place ',
      e('code', { style: { fontFamily: 'JetBrains Mono', background: '#00000033', borderRadius: 4, padding: '0 4px' } }, 'pcb_model.pt'),
      ' in the ',
      e('code', { style: { fontFamily: 'JetBrains Mono', background: '#00000033', borderRadius: 4, padding: '0 4px' } }, 'models/'),
      ' directory.'
    )
  );
}

function DefectList({ detections }) {
  if (!detections || detections.length === 0) {
    return e('div', { className: 'empty-state' },
      e('span', { className: 'empty-icon' }, '✅'),
      e('div', { className: 'empty-text' }, 'No defects detected\nin this image')
    );
  }

  // Group by class
  const grouped = {};
  detections.forEach(d => {
    const k = d.class_name.toLowerCase().replace(/ /g, '_');
    if (!grouped[k]) grouped[k] = { count: 0, confs: [], color: classColor(d.class_name) };
    grouped[k].count++;
    grouped[k].confs.push(d.confidence);
  });

  return e('div', null,
    ...Object.entries(grouped).map(([klass, info]) => {
      const avgConf = info.confs.reduce((a, b) => a + b, 0) / info.confs.length;
      return e('div', { key: klass, className: 'defect-item' },
        e('span', { className: 'defect-dot', style: { background: info.color } }),
        e('div', { style: { flex: 1 } },
          e('div', { style: { display: 'flex', alignItems: 'center', justifyContent: 'space-between' } },
            e('span', { className: 'defect-name' }, klass.replace(/_/g, ' ')),
            e('div', { style: { display: 'flex', alignItems: 'center', gap: 8 } },
              e('span', { className: 'defect-conf' }, (avgConf * 100).toFixed(0) + '%'),
              e(Badge, { count: info.count, color: info.color })
            )
          ),
          e('div', { className: 'conf-bar-wrap', style: { marginTop: 6 } },
            e('div', { className: 'conf-bar-fill', style: { width: (avgConf * 100) + '%', background: info.color } })
          )
        )
      );
    })
  );
}

// ---------- Upload Panel ---------- //

function UploadPanel({ modelStatus }) {
  const [resultUrl, setResultUrl] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [status, setStatus] = useState('idle'); // idle | loading | done | error
  const [statusMsg, setStatusMsg] = useState('Ready to analyse');
  const [detections, setDetections] = useState(null);
  const [dragOver, setDragOver] = useState(false);
  const [totalScanned, setTotalScanned] = useState(0);
  const [totalDefects, setTotalDefects] = useState(0);

  async function runDetect(file) {
    if (!file) return;
    setStatus('loading');
    setStatusMsg('Analysing image…');
    setResultUrl(null);
    setDetections(null);

    // Show preview immediately
    const reader = new FileReader();
    reader.onload = ev => setPreviewUrl(ev.target.result);
    reader.readAsDataURL(file);

    const form = new FormData();
    form.append('file', file, file.name);

    try {
      const res = await fetch('/api/detect-upload', { method: 'POST', body: form });
      if (!res.ok) {
        let j = null;
        try { j = await res.json(); } catch { }
        throw new Error((j && (j.detail || j.error)) || `HTTP ${res.status}`);
      }

      // Try to read detection metadata from header
      const metaHeader = res.headers.get('x-detection-meta');
      let meta = null;
      if (metaHeader) {
        try { meta = JSON.parse(metaHeader); } catch { }
      }

      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      setResultUrl(url);
      setDetections(meta ? meta.detections : []);
      setStatus('done');
      setStatusMsg(`Detection complete — ${meta ? meta.count : '?'} object(s) found`);
      setTotalScanned(prev => prev + 1);
      setTotalDefects(prev => prev + (meta ? meta.count : 0));
    } catch (err) {
      setStatus('error');
      setStatusMsg('Error: ' + err.message);
    }
  }

  function handleFileChange(ev) { runDetect(ev.target.files[0]); }
  function handleDrop(ev) {
    ev.preventDefault(); setDragOver(false);
    const file = ev.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) runDetect(file);
  }

  const statusDotClass = { idle: 'idle', loading: 'running', done: 'done', error: 'error' }[status];

  return e('div', null,
    !modelStatus.has_pcb_model && e(NoModelBanner),

    e('div', { className: 'main-grid' },
      // Left: drop zone + result
      e('div', null,
        e('div', { className: 'panel-card' },
          e('div', { className: 'panel-header' },
            e('span', { className: 'icon' }, '🔍'),
            'PCB Image Analysis'
          ),
          e('div', { className: 'panel-body' },
            e('div', {
              className: `drop-zone ${dragOver ? 'drag-over' : ''}`,
              onDragOver: ev => { ev.preventDefault(); setDragOver(true); },
              onDragLeave: () => setDragOver(false),
              onDrop: handleDrop,
            },
              e('input', { type: 'file', accept: 'image/*', onChange: handleFileChange }),
              e('span', { className: 'drop-icon' }, '🖼️'),
              e('div', { className: 'drop-title' }, 'Drop PCB image here'),
              e('div', { className: 'drop-sub' }, 'or click to browse — JPG, PNG, BMP, TIFF')
            ),

            status !== 'idle' && e('div', { className: 'status-bar' },
              e('span', { className: `status-dot ${statusDotClass}` }),
              statusMsg
            ),

            (resultUrl || (status === 'loading' && previewUrl)) && e('div', { className: 'result-wrapper' },
              e('img', { src: resultUrl || previewUrl, alt: 'Result' }),
              status === 'loading' && e('div', { className: 'analyzing-overlay' },
                e('div', { className: 'spinner' }),
                e('div', { className: 'analyzing-text' }, 'Running AI inference…')
              ),
              status === 'done' && detections !== null && e('div', { className: 'result-overlay-bar' },
                e('span', { className: 'result-label' }, '🎯 Detection Result'),
                e('span', { className: 'result-count-badge' }, `${detections.length} detection${detections.length !== 1 ? 's' : ''}`)
              )
            )
          )
        )
      ),

      // Right: results sidebar
      e('div', { className: 'results-panel' },
        e('div', { className: 'stats-bar', style: { gridTemplateColumns: '1fr 1fr 1fr', marginBottom: 0 } },
          e(StatCard, { value: totalScanned, label: 'Scanned' }),
          e(StatCard, { value: totalDefects, label: 'Defects' }),
          e(StatCard, { value: detections !== null ? detections.length : '—', label: 'This image' })
        ),

        e('div', { className: 'panel-card defect-list-card' },
          e('div', { className: 'panel-header' },
            e('span', { className: 'icon' }, '📊'),
            'Defect Breakdown'
          ),
          e('div', { className: 'panel-body', style: { padding: 0 } },
            status === 'loading'
              ? e('div', { className: 'empty-state' },
                e('div', { className: 'spinner', style: { margin: '0 auto' } })
              )
              : e(DefectList, { detections: detections || [] })
          )
        ),

        e(ClassLegend)
      )
    )
  );
}

// ---------- Webcam Panel ---------- //

function WebcamPanel({ modelStatus }) {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [streaming, setStreaming] = useState(false);
  const [resultUrl, setResultUrl] = useState(null);
  const [detections, setDetections] = useState([]);
  const streamingRef = useRef(false);

  async function startStream() {
    if (streaming) return;
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      videoRef.current.srcObject = stream;
      await videoRef.current.play();
      setStreaming(true);
      streamingRef.current = true;
      requestAnimationFrame(loop);
    } catch (err) {
      alert('Camera error: ' + err.message);
    }
  }

  function stopStream() {
    streamingRef.current = false;
    const stream = videoRef.current && videoRef.current.srcObject;
    if (stream) stream.getTracks().forEach(t => t.stop());
    if (videoRef.current) videoRef.current.srcObject = null;
    setStreaming(false);
  }

  async function loop() {
    if (!streamingRef.current) return;
    const v = videoRef.current;
    const c = canvasRef.current;
    if (v && c && v.videoWidth) {
      c.width = v.videoWidth; c.height = v.videoHeight;
      c.getContext('2d').drawImage(v, 0, 0);
      c.toBlob(async blob => {
        try {
          const res = await fetch('/api/detect-frame', { method: 'POST', body: blob });
          if (res.ok) {
            const metaHeader = res.headers.get('x-detection-meta');
            if (metaHeader) {
              try { setDetections(JSON.parse(metaHeader).detections || []); } catch { }
            }
            const b = await res.blob();
            setResultUrl(prev => { if (prev) URL.revokeObjectURL(prev); return URL.createObjectURL(b); });
          }
        } catch { }
        setTimeout(() => requestAnimationFrame(loop), 800);
      }, 'image/jpeg', 0.75);
    } else {
      setTimeout(() => requestAnimationFrame(loop), 200);
    }
  }

  return e('div', null,
    !modelStatus.has_pcb_model && e(NoModelBanner),
    e('div', { className: 'main-grid' },
      e('div', null,
        e('div', { className: 'panel-card webcam-panel' },
          e('div', { className: 'panel-header' },
            e('span', { className: 'icon' }, '📷'),
            'Live Webcam Detection'
          ),
          e('div', { className: 'panel-body' },
            e('div', { className: 'webcam-stream-row' },
              e('div', { className: 'stream-box' },
                e('span', { className: 'stream-label' }, 'Camera'),
                e('video', { ref: videoRef, autoPlay: true, playsInline: true, muted: true }),
              ),
              e('div', { className: 'stream-box' },
                e('span', { className: 'stream-label' }, 'Annotated'),
                resultUrl
                  ? e('img', { src: resultUrl, alt: 'Annotated' })
                  : e('div', { className: 'stream-placeholder' }, '🎯\nStart camera to see\nannotated output')
              )
            ),
            e('canvas', { ref: canvasRef, style: { display: 'none' } }),
            e('div', { className: 'webcam-btns' },
              e('button', { className: 'btn btn-primary', onClick: startStream, disabled: streaming },
                streaming ? '📡 Streaming…' : '▶ Start Camera'
              ),
              streaming && e('button', { className: 'btn btn-danger', onClick: stopStream }, '■ Stop')
            )
          )
        )
      ),

      e('div', { className: 'results-panel' },
        e('div', { className: 'panel-card defect-list-card' },
          e('div', { className: 'panel-header' },
            e('span', { className: 'icon' }, '📊'),
            'Live Detections'
          ),
          e('div', { className: 'panel-body', style: { padding: 0 } },
            e(DefectList, { detections })
          )
        ),
        e(ClassLegend)
      )
    )
  );
}

// ---------- Root App ---------- //

function App() {
  const [tab, setTab] = useState('upload');
  const [modelStatus, setModelStatus] = useState({ has_pcb_model: true, model_name: 'Loading…' });

  useEffect(() => {
    fetch('/api/model-status')
      .then(r => r.json())
      .then(data => setModelStatus(data))
      .catch(() => setModelStatus({ has_pcb_model: false, model_name: 'Unknown' }));
  }, []);

  return e('div', { className: 'app-wrapper' },
    e('header', { className: 'header' },
      e('div', { className: 'header-badge' },
        e('span', { className: 'pulse-dot' }),
        'AI-Powered PCB Inspection System'
      ),
      e('h1', null, 'PCB ', e('span', null, 'Defect'), ' Detection'),
      e('p', null, 'Upload PCB images or use your webcam for real-time automated defect classification using YOLO11.')
    ),

    e('div', { style: { display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 20 } },
      e('div', { className: 'tab-controls' },
        e('button', { className: `tab-btn ${tab === 'upload' ? 'active' : ''}`, onClick: () => setTab('upload') }, '🖼️ Upload Image'),
        e('button', { className: `tab-btn ${tab === 'webcam' ? 'active' : ''}`, onClick: () => setTab('webcam') }, '📷 Webcam')
      ),
      e('div', { className: 'status-bar', style: { marginTop: 0, fontSize: 12 } },
        e('span', { className: `status-dot ${modelStatus.has_pcb_model ? 'done' : 'error'}` }),
        e('span', null, 'Model: '),
        e('code', { style: { fontFamily: 'JetBrains Mono', fontSize: 11 } }, modelStatus.model_name || '—')
      )
    ),

    tab === 'upload'
      ? e(UploadPanel, { modelStatus })
      : e(WebcamPanel, { modelStatus }),

    e('footer', { className: 'footer' },
      'PCB Defect Detection • Powered by YOLO11 & FastAPI • 6 defect classes: missing hole, mouse bite, open circuit, short, spur, spurious copper'
    )
  );
}

ReactDOM.createRoot(document.getElementById('root')).render(e(App));
