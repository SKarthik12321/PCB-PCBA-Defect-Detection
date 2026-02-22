import { useState, useRef } from 'react'
import { Camera, VideoOff, Play, Square, BarChart2 } from 'lucide-react'
import DefectList from './DefectList'
import ClassLegend from './ClassLegend'

export default function WebcamPanel({ modelStatus }) {
    const videoRef = useRef(null)
    const canvasRef = useRef(null)
    const streamingRef = useRef(false)

    const [streaming, setStreaming] = useState(false)
    const [resultUrl, setResultUrl] = useState(null)
    const [detections, setDetections] = useState([])

    async function startStream() {
        if (streaming) return
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ video: true })
            videoRef.current.srcObject = stream
            await videoRef.current.play()
            setStreaming(true)
            streamingRef.current = true
            requestAnimationFrame(loop)
        } catch (err) {
            alert('Camera error: ' + err.message)
        }
    }

    function stopStream() {
        streamingRef.current = false
        const stream = videoRef.current && videoRef.current.srcObject
        if (stream) stream.getTracks().forEach(t => t.stop())
        if (videoRef.current) videoRef.current.srcObject = null
        setStreaming(false)
    }

    async function loop() {
        if (!streamingRef.current) return
        const v = videoRef.current
        const c = canvasRef.current
        if (v && c && v.videoWidth) {
            c.width = v.videoWidth
            c.height = v.videoHeight
            c.getContext('2d').drawImage(v, 0, 0)
            c.toBlob(async blob => {
                try {
                    const apiUrl = import.meta.env.VITE_API_URL || '';
                    const res = await fetch(`${apiUrl}/api/detect-frame`, { method: 'POST', body: blob })
                    if (res.ok) {
                        const metaHeader = res.headers.get('x-detection-meta')
                        if (metaHeader) {
                            try { setDetections(JSON.parse(metaHeader).detections || []) } catch { }
                        }
                        const b = await res.blob()
                        setResultUrl(prev => {
                            if (prev) URL.revokeObjectURL(prev)
                            return URL.createObjectURL(b)
                        })
                    }
                } catch { }
                setTimeout(() => requestAnimationFrame(loop), 800)
            }, 'image/jpeg', 0.75)
        } else {
            setTimeout(() => requestAnimationFrame(loop), 200)
        }
    }

    return (
        <div>
            <div className="main-grid">
                {/* ── Left: webcam streams ── */}
                <div>
                    <div className="card">
                        <div className="card-header">
                            <span className="card-header-icon">
                                <Camera size={14} />
                            </span>
                            Live Webcam Detection
                        </div>
                        <div className="card-body" style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
                            <div className="webcam-stream-row">
                                <div className="stream-box">
                                    <span className="stream-label">Camera</span>
                                    <video ref={videoRef} autoPlay playsInline muted />
                                </div>
                                <div className="stream-box">
                                    <span className="stream-label">Annotated</span>
                                    {resultUrl
                                        ? (
                                            <div style={{ position: 'relative', width: '100%', height: '100%' }}>
                                                <img src={resultUrl} alt="Annotated" />
                                                <div className="live-inference-badge">
                                                    <span className="dot pulse" /> Live Inference
                                                </div>
                                            </div>
                                        )
                                        : (
                                            <div className="stream-placeholder">
                                                <VideoOff size={24} strokeWidth={1.5} style={{ opacity: 0.35 }} />
                                                <span>Start camera to see<br />annotated output</span>
                                            </div>
                                        )
                                    }
                                </div>
                            </div>

                            <canvas ref={canvasRef} style={{ display: 'none' }} />

                            <div className="webcam-actions">
                                <button
                                    className="btn btn-primary"
                                    onClick={startStream}
                                    disabled={streaming}
                                >
                                    <Play size={14} />
                                    {streaming ? 'Streaming...' : 'Start Camera'}
                                </button>
                                {streaming && (
                                    <button className="btn btn-danger" onClick={stopStream}>
                                        <Square size={14} />
                                        Stop
                                    </button>
                                )}
                            </div>
                        </div>
                    </div>
                </div>

                {/* ── Right: live detections ── */}
                <div className="results-panel">
                    <div className="card defect-list-card">
                        <div className="card-header">
                            <span className="card-header-icon">
                                <BarChart2 size={14} />
                            </span>
                            Live Detections
                        </div>
                        <div className="card-body no-pad">
                            <DefectList detections={detections} />
                        </div>
                    </div>

                    <ClassLegend />
                </div>
            </div>
        </div>
    )
}
