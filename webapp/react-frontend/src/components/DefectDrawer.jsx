import { useState } from 'react'
import { X, ChevronRight, Zap, Circle, GitBranch, Minus, Waves, Cpu } from 'lucide-react'

const DEFECTS = [
    {
        key: 'missing_hole',
        label: 'Missing Hole',
        color: '#ef4444',
        Icon: Circle,
        severity: 'Critical',
        sevColor: '#ef4444',
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
        sevColor: '#f97316',
        what: 'A series of small irregular notches along the PCB edge, caused by incomplete board separation.',
        cause: 'V-scoring or tab-routing cutter wear, excessive board panel stress during depanelization, or worn tooling.',
        impact: 'Weakens mechanical integrity of the board edge and can break nearby traces, causing intermittent circuit failures.',
    },
    {
        key: 'open_circuit',
        label: 'Open Circuit',
        color: '#eab308',
        Icon: Minus,
        severity: 'Critical',
        sevColor: '#ef4444',
        what: 'A copper trace is physically broken or severed, creating an incomplete electrical path.',
        cause: 'Over-etching during chemical processing, mechanical scratching, copper delamination, or thermal stress cracking during reflow.',
        impact: 'Signals cannot pass through the affected net. The circuit is non-functional at that node.',
    },
    {
        key: 'short',
        label: 'Short Circuit',
        color: '#a855f7',
        Icon: Zap,
        severity: 'Critical',
        sevColor: '#ef4444',
        what: 'Two copper traces or pads that should be isolated are unintentionally connected.',
        cause: 'Insufficient etching leaving copper bridges, solder bridges during reflow, or contamination during manufacturing.',
        impact: 'Causes excessive current draw, component damage, power supply shutdown, or catastrophic board failure.',
    },
    {
        key: 'spur',
        label: 'Spur',
        color: '#06b6d4',
        Icon: GitBranch,
        severity: 'Medium',
        sevColor: '#eab308',
        what: 'An unintended thin copper protrusion extending from a trace or pad into the surrounding area.',
        cause: 'Photoresist pinholes during lithography, under-etching that leaves extra copper, or mask adhesion failures.',
        impact: 'Reduces clearance between nets, increasing risk of shorts under vibration. May also affect impedance.',
    },
    {
        key: 'spurious_copper',
        label: 'Spurious Copper',
        color: '#22c55e',
        Icon: Cpu,
        severity: 'Medium',
        sevColor: '#eab308',
        what: 'Isolated patches of copper appearing where no copper should exist on the board.',
        cause: 'Inadequate etching, artwork generation errors, or residual plating from electroplating that was not removed.',
        impact: 'Creates unintended conductive paths that may short-circuit nearby traces or degrade signal integrity.',
    },
]

function DefectRow({ defect, isOpen, onToggle }) {
    const { label, color, Icon, severity, sevColor, what, cause, impact } = defect
    return (
        <div className="dr-row" style={{ borderColor: isOpen ? color + '44' : undefined }}>
            <button className="dr-row-btn" onClick={onToggle}>
                <span className="dr-dot" style={{ background: color, boxShadow: `0 0 5px ${color}66` }} />
                <span className="dr-icon" style={{ color, background: color + '15' }}>
                    <Icon size={13} strokeWidth={1.75} />
                </span>
                <span className="dr-label">{label}</span>
                <span className="dr-sev" style={{ color: sevColor, background: sevColor + '15', border: `1px solid ${sevColor}33` }}>
                    {severity}
                </span>
                <ChevronRight size={14} className={`dr-chevron${isOpen ? ' dr-chevron--open' : ''}`} style={{ color }} />
            </button>
            {isOpen && (
                <div className="dr-body">
                    <div className="dr-cols">
                        <div>
                            <div className="dr-col-label">What is it?</div>
                            <p className="dr-col-text">{what}</p>
                        </div>
                        <div>
                            <div className="dr-col-label">Root Cause</div>
                            <p className="dr-col-text">{cause}</p>
                        </div>
                        <div>
                            <div className="dr-col-label">Impact</div>
                            <p className="dr-col-text">{impact}</p>
                        </div>
                    </div>
                </div>
            )}
        </div>
    )
}

export default function DefectDrawer({ open, onClose }) {
    const [openKey, setOpenKey] = useState(null)

    function toggle(key) {
        setOpenKey(prev => prev === key ? null : key)
    }

    return (
        <>
            {/* Backdrop */}
            {open && <div className="drawer-backdrop" onClick={onClose} />}

            {/* Drawer panel */}
            <aside className={`drawer${open ? ' drawer--open' : ''}`}>
                <div className="drawer-head">
                    <div>
                        <div className="drawer-title">Defect Reference</div>
                        <div className="drawer-sub">Click a defect to see definition, cause &amp; impact</div>
                    </div>
                    <button className="drawer-close" onClick={onClose} aria-label="Close">
                        <X size={16} />
                    </button>
                </div>

                <div className="drawer-body">
                    {DEFECTS.map(d => (
                        <DefectRow
                            key={d.key}
                            defect={d}
                            isOpen={openKey === d.key}
                            onToggle={() => toggle(d.key)}
                        />
                    ))}
                </div>
            </aside>
        </>
    )
}
