import { CheckCircle, AlertCircle, BarChart2 } from 'lucide-react'

const CLASS_COLORS = {
    missing_hole: '#ef4444',
    mouse_bite: '#f97316',
    open_circuit: '#eab308',
    short: '#a855f7',
    spur: '#06b6d4',
    spurious_copper: '#22c55e',
}

function classColor(name) {
    const k = name.toLowerCase().replace(/ /g, '_')
    return CLASS_COLORS[k] || '#94a3b8'
}

export default function DefectList({ detections }) {
    if (!detections || detections.length === 0) {
        return (
            <div className="empty-state">
                <CheckCircle size={32} className="empty-state-icon" strokeWidth={1.5} />
                <p className="empty-text">No defects detected in this image</p>
            </div>
        )
    }

    // Group by class
    const grouped = {}
    detections.forEach(d => {
        const k = d.class_name.toLowerCase().replace(/ /g, '_')
        if (!grouped[k]) {
            grouped[k] = { count: 0, confs: [], color: classColor(d.class_name) }
        }
        grouped[k].count++
        grouped[k].confs.push(d.confidence)
    })

    return (
        <div>
            {Object.entries(grouped).map(([klass, info]) => {
                const avgConf = info.confs.reduce((a, b) => a + b, 0) / info.confs.length
                return (
                    <div key={klass} className="defect-item">
                        <span className="defect-dot" style={{ background: info.color }} />
                        <div style={{ flex: 1, minWidth: 0 }}>
                            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                                <span className="defect-name">{klass.replace(/_/g, ' ')}</span>
                                <div className="defect-meta">
                                    <span className="defect-conf">{(avgConf * 100).toFixed(0)}%</span>
                                    <span
                                        className="defect-badge"
                                        style={{
                                            background: info.color + '20',
                                            color: info.color,
                                            border: `1px solid ${info.color}44`,
                                        }}
                                    >
                                        {info.count}
                                    </span>
                                </div>
                            </div>
                            <div className="conf-bar">
                                <div
                                    className="conf-bar-fill"
                                    style={{ width: `${avgConf * 100}%`, background: info.color }}
                                />
                            </div>
                        </div>
                    </div>
                )
            })}
        </div>
    )
}
