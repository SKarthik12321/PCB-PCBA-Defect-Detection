import { useState, useEffect } from 'react'
import { Upload, Camera, Cpu, BookOpen, RefreshCw } from 'lucide-react'
import Header from './components/Header'
import UploadPanel from './components/UploadPanel'
import WebcamPanel from './components/WebcamPanel'
import DefectDrawer from './components/DefectDrawer'
import PCBBackground from './components/PCBBackground'

export default function App() {
  const [tab, setTab] = useState('upload')
  const [drawerOpen, setDrawerOpen] = useState(false)
  const [modelStatus, setModelStatus] = useState({
    has_pcb_model: true,
    model_name: 'Loading...',
    ready: false,
  })

  useEffect(() => {
    fetch('/api/model-status')
      .then(r => r.json())
      .then(data => setModelStatus(data))
      .catch(() => setModelStatus({ has_pcb_model: false, model_name: 'Unavailable', ready: false }))
  }, [])

  return (
    <div className="app-wrapper">
      <PCBBackground />
      <Header />

      {/* ── Toolbar ── */}
      <div className="toolbar">
        <div className="tab-controls">
          <button
            className={`tab-btn${tab === 'upload' ? ' active' : ''}`}
            onClick={() => setTab('upload')}
          >
            <Upload size={14} />
            Upload Image
          </button>
          <button
            className={`tab-btn${tab === 'webcam' ? ' active' : ''}`}
            onClick={() => setTab('webcam')}
          >
            <Camera size={14} />
            Webcam
          </button>
          <button
            className="tab-btn tab-btn--learn"
            onClick={() => setDrawerOpen(true)}
          >
            <BookOpen size={14} />
            Learn Defects
          </button>
        </div>

        <div style={{ display: 'flex', gap: '12px', alignItems: 'center' }}>
          <button
            className="tab-btn"
            style={{ border: '1px solid var(--border)', background: 'rgba(12,18,30,0.75)', backdropFilter: 'blur(12px)', color: 'var(--text-secondary)' }}
            onClick={() => window.location.reload()}
          >
            <RefreshCw size={14} />
            Refresh & Clean Cache
          </button>

          <div className="model-badge">
            <Cpu size={12} />
            <span>Model:</span>
            <code>{modelStatus.model_name || '—'}</code>
            <span
              className={`dot ${modelStatus.ready
                ? modelStatus.has_pcb_model ? 'ok' : 'idle'
                : 'pulse'
                }`}
            />
          </div>
        </div>
      </div>

      {/* ── Panel ── */}
      {tab === 'upload'
        ? <UploadPanel modelStatus={modelStatus} />
        : <WebcamPanel modelStatus={modelStatus} />
      }

      {/* ── Defect Drawer ── */}
      <DefectDrawer open={drawerOpen} onClose={() => setDrawerOpen(false)} />
    </div>
  )
}
