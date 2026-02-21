import { useState } from 'react'
import { ChevronDown, ChevronRight, Zap, Circle, GitBranch, Minus, Waves, Cpu } from 'lucide-react'

const DEFECTS = [
    {
        key: 'missing_hole',
        label: 'Missing Hole',
        color: '#ef4444',
        Icon: Circle,
        severity: 'Critical',
        what: 'A through-hole or via that was supposed to be drilled is completely absent from the PCB.',
        cause: 'Mechanical drill bit breakage, misaligned drill files, CNC programming error, or toolpath offset miscalibration during fabrication.',
        impact: 'Component cannot be inserted or soldered. Vias connecting board layers are non-functional, breaking signal or power routing.',
    },
    {
        key: 'mouse_bite',
        label: 'Mouse Bite',
        color: '#f97316',
        Icon: Waves,
        severity: 'High',
        what: 'A series of small, irregular notches along the PCB edge — resembling bite marks — caused by incomplete board separation.',
        cause: 'V-scoring or tab-routing cutter wear, excessive board panel stress during depanelization, or worn tooling that tears copper traces near the edge.',
        impact: 'Weakens the mechanical integrity of the board edge and can break nearby traces, causing intermittent or permanent circuit failures.',
    },
    {
        key: 'open_circuit',
        label: 'Open Circuit',
        color: '#eab308',
        Icon: Minus,
        severity: 'Critical',
        what: 'A copper trace is physically broken or severed, creating an incomplete electrical path between two points.',
        cause: 'Over-etching during chemical processing, mechanical scratching, copper delamination, or thermal stress cracking during reflow soldering.',
        impact: 'Signals cannot pass through the affected net. The circuit is non-functional at that node — equivalent to a disconnected wire.',
    },
    {
        key: 'short',
        label: 'Short Circuit',
        color: '#a855f7',
        Icon: Zap,
        severity: 'Critical',
        what: 'Two copper traces or pads that should be electrically isolated are unintentionally connected.',
        cause: 'Insufficient etching leaving copper bridges between traces, solder bridges during wave or reflow soldering, or contamination during manufacturing.',
        impact: 'Can cause excessive current draw, component damage, power supply shutdown, or catastrophic board failure. Often the most dangerous defect class.',
    },
    {
        key: 'spur',
        label: 'Spur',
        color: '#06b6d4',
        Icon: GitBranch,
        severity: 'Medium',
        what: 'An unintended thin copper protrusion extending from a trace or pad into the surrounding area.',
        cause: 'Photoresist pinholes during lithography, under-etching that leaves extra copper, or mask adhesion failures creating unwanted copper islands.',
        impact: 'Can reduce clearance between adjacent nets, increasing risk of short circuits under vibration or thermal expansion. May also affect impedance characteristics.',
    },
    {
        key: 'spurious_copper',
        label: 'Spurious Copper',
        color: '#22c55e',
        Icon: Cpu,
        severity: 'Medium',
        what: 'Isolated patches of copper that appear in areas of the PCB where no copper should exist.',
        cause: 'Inadequate etching of unwanted copper, artwork generation errors, or residual plating from electroplating that was not removed.',
        impact: 'Creates unintended conductive paths that may short-circuit nearby traces or interfere with signal integrity and RF performance.',
    },
]

const SEVERITY_STYLE = {
    Critical: { color: '#ef4444', bg: 'rgba(239,68,68,0.12)', border: 'rgba(239,68,68,0.25)' },
    High: { color: '#f97316', bg: 'rgba(249,115,22,0.12)', border: 'rgba(249,115,22,0.25)' },
    Medium: { color: '#eab308', bg: 'rgba(234,179,8,0.10)', border: 'rgba(234,179,8,0.22)' },
}

function DefectCard({ defect, isOpen, onToggle }) {
    const { label, color, Icon, severity, what, cause, impact } = defect
    const sev = SEVERITY_STYLE[severity]

    return (
        <div
            className={`defect-card${isOpen ? ' defect-card--open' : ''}`}
            style={{ borderColor: isOpen ? color + '55' : undefined }}
        >
            <button className="defect-card-header" onClick={onToggle}>
                <div className="defect-card-left">
                    <span className="defect-card-dot" style={{ background: color, boxShadow: `0 0 6px ${color}66` }} />
                    <span className="defect-card-icon" style={{ color, background: color + '18' }}>
                        <Icon size={14} strokeWidth={1.75} />
                    </span>
                    <span className="defect-card-label">{label}</span>
                    <span
                        className="defect-severity-badge"
                        style={{ color: sev.color, background: sev.bg, border: `1px solid ${sev.border}` }}
                    >
                        {severity}
                    </span>
                </div>
                <span className="defect-card-chevron" style={{ color }}>
                    {isOpen ? <ChevronDown size={15} /> : <ChevronRight size={15} />}
                </span>
            </button>

            {isOpen && (
                <div className="defect-card-body">
                    <div className="defect-info-row">
                        <div className="defect-info-block">
                            <div className="defect-info-label">What is it?</div>
                            <p className="defect-info-text">{what}</p>
                        </div>
                        <div className="defect-info-block">
                            <div className="defect-info-label">Root Cause</div>
                            <p className="defect-info-text">{cause}</p>
                        </div>
                        <div className="defect-info-block">
                            <div className="defect-info-label">Impact</div>
                            <p className="defect-info-text">{impact}</p>
                        </div>
                    </div>
                </div>
            )}
        </div>
    )
}

export default function DefectEncyclopedia() {
    const [openKey, setOpenKey] = useState(null)

    function toggle(key) {
        setOpenKey(prev => prev === key ? null : key)
    }

    return (
        <section className="encyclopedia-section">
            <div className="encyclopedia-header">
                <div className="encyclopedia-title-block">
                    <h2 className="encyclopedia-title">Defect Reference</h2>
                    <p className="encyclopedia-subtitle">
                        Click any defect to learn its definition, root cause, and board impact.
                    </p>
                </div>
                <div className="encyclopedia-count">
                    {DEFECTS.length} defect classes
                </div>
            </div>

            <div className="defect-cards-list">
                {DEFECTS.map(d => (
                    <DefectCard
                        key={d.key}
                        defect={d}
                        isOpen={openKey === d.key}
                        onToggle={() => toggle(d.key)}
                    />
                ))}
            </div>
        </section>
    )
}
