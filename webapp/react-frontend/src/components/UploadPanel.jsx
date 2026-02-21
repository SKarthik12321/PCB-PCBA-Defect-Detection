import { useState, useRef } from 'react'
import { Upload, Image as ImageIcon, BarChart2 } from 'lucide-react'
import StatCard from './StatCard'
import DefectList from './DefectList'
import ClassLegend from './ClassLegend'

export default function UploadPanel({ modelStatus }) {
    const [resultUrl, setResultUrl] = useState(null)
    const [previewUrl, setPreviewUrl] = useState(null)
    const [status, setStatus] = useState('idle')   // idle|loading|done|error
    const [statusMsg, setStatusMsg] = useState('Ready to analyse')
    const [detections, setDetections] = useState(null)
    const [dragOver, setDragOver] = useState(false)
    const [totalScanned, setTotalScanned] = useState(0)
    const [totalDefects, setTotalDefects] = useState(0)

    async function runDetect(file) {
        if (!file) return
        setStatus('loading')
        setStatusMsg('Analysing image...')
        setResultUrl(null)
        setDetections(null)

        const reader = new FileReader()
        reader.onload = ev => setPreviewUrl(ev.target.result)
        reader.readAsDataURL(file)

        const form = new FormData()
        form.append('file', file, file.name)

        try {
            const res = await fetch('/api/detect-upload', { method: 'POST', body: form })
            if (!res.ok) {
                let j = null
                try { j = await res.json() } catch { }
                throw new Error((j && (j.detail || j.error)) || `HTTP ${res.status}`)
            }

            const metaHeader = res.headers.get('x-detection-meta')
            let meta = null
            if (metaHeader) {
                try { meta = JSON.parse(metaHeader) } catch { }
            }

            const blob = await res.blob()
            const url = URL.createObjectURL(blob)
            setResultUrl(url)
            setDetections(meta ? meta.detections : [])
            setStatus('done')
            setStatusMsg(`Detection complete — ${meta ? meta.count : '?'} object(s) found`)
            setTotalScanned(p => p + 1)
            setTotalDefects(p => p + (meta ? meta.count : 0))
        } catch (err) {
            setStatus('error')
            setStatusMsg('Error: ' + err.message)
        }
    }

    function handleFileChange(ev) { runDetect(ev.target.files[0]) }

    function handleDrop(ev) {
        ev.preventDefault()
        setDragOver(false)
        const file = ev.dataTransfer.files[0]
        if (file && file.type.startsWith('image/')) runDetect(file)
    }

    const dotClass = { idle: 'idle', loading: 'pulse', done: 'ok', error: 'err' }[status]

    return (
        <div>
            <div className="main-grid">
                {/* ── Left: image panel ── */}
                <div>
                    <div className="card">
                        <div className="card-header">
                            <span className="card-header-icon">
                                <ImageIcon size={14} />
                            </span>
                            PCB Image Analysis
                        </div>
                        <div className="card-body">
                            <div
                                className={`drop-zone${dragOver ? ' drag-over' : ''}`}
                                onDragOver={ev => { ev.preventDefault(); setDragOver(true) }}
                                onDragLeave={() => setDragOver(false)}
                                onDrop={handleDrop}
                            >
                                <input
                                    type="file"
                                    accept="image/*"
                                    onChange={handleFileChange}
                                />
                                <div className="drop-icon">
                                    <Upload size={22} strokeWidth={1.75} />
                                </div>
                                <div className="drop-title">Drop PCB image here</div>
                                <div className="drop-sub">or click to browse — JPG, PNG, BMP, TIFF</div>
                            </div>

                            {status !== 'idle' && (
                                <div className="status-bar">
                                    <span className={`dot ${dotClass}`} />
                                    {statusMsg}
                                </div>
                            )}

                            {(resultUrl || (status === 'loading' && previewUrl)) && (
                                <div className="result-wrapper">
                                    <img src={resultUrl || previewUrl} alt="Result" />
                                    {status === 'loading' && (
                                        <div className="analyzing-overlay">
                                            <div className="spinner" />
                                            <div className="analyzing-text">Running AI inference...</div>
                                        </div>
                                    )}
                                    {status === 'done' && detections !== null && (
                                        <div className="result-overlay-bar">
                                            <span className="result-label">Detection Result</span>
                                            <span className="result-count-badge">
                                                {detections.length} detection{detections.length !== 1 ? 's' : ''}
                                            </span>
                                        </div>
                                    )}
                                </div>
                            )}
                        </div>
                    </div>
                </div>

                {/* ── Right: results sidebar ── */}
                <div className="results-panel">
                    <div className="stats-row">
                        <StatCard value={totalScanned} label="Scanned" />
                        <StatCard value={totalDefects} label="Defects" />
                        <StatCard value={detections !== null ? detections.length : '—'} label="This image" />
                    </div>

                    <div className="card defect-list-card">
                        <div className="card-header">
                            <span className="card-header-icon">
                                <BarChart2 size={14} />
                            </span>
                            Defect Breakdown
                        </div>
                        <div className="card-body no-pad">
                            {status === 'loading'
                                ? <div className="empty-state"><div className="spinner" /></div>
                                : <DefectList detections={detections || []} />
                            }
                        </div>
                    </div>

                    <ClassLegend />
                </div>
            </div>
        </div>
    )
}
