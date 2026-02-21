import { Scan } from 'lucide-react'

export default function Header() {
    return (
        <header className="header">
            <div className="header-badge">
                <span className="live-dot" />
                PCB/PCBA Inspection System
            </div>

            <h1>
                <div className="header-title-row">
                    <Scan size={36} className="header-icon" strokeWidth={1.5} />
                    PCB <span>Defect</span> Detection
                </div>
            </h1>

            <p>
                Upload PCB images or use your webcam for automated real-time
                defect classification powered by YOLO11.
            </p>
        </header>
    )
}
