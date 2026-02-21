import { Layers } from 'lucide-react'

const CLASS_META = [
    { key: 'missing_hole', label: 'Missing Hole', color: '#ef4444' },
    { key: 'mouse_bite', label: 'Mouse Bite', color: '#f97316' },
    { key: 'open_circuit', label: 'Open Circuit', color: '#eab308' },
    { key: 'short', label: 'Short', color: '#a855f7' },
    { key: 'spur', label: 'Spur', color: '#06b6d4' },
    { key: 'spurious_copper', label: 'Spurious Copper', color: '#22c55e' },
]

export default function ClassLegend() {
    return (
        <div className="class-legend">
            <div className="class-legend-title">
                <Layers size={11} style={{ display: 'inline', marginRight: 5, verticalAlign: 'middle' }} />
                Defect Classes
            </div>
            <div className="class-chips">
                {CLASS_META.map(c => (
                    <span
                        key={c.key}
                        className="class-chip"
                        style={{
                            color: c.color,
                            borderColor: c.color + '44',
                            background: c.color + '12',
                        }}
                    >
                        <span className="chip-dot" style={{ background: c.color }} />
                        {c.label}
                    </span>
                ))}
            </div>
        </div>
    )
}
