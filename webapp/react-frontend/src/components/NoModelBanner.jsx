import { AlertTriangle, X, Info } from 'lucide-react'
import { useState } from 'react'

export default function NoModelBanner() {
    const [dismissed, setDismissed] = useState(false)
    if (dismissed) return null

    return (
        <div className="model-notice">
            <div className="model-notice-left">
                <span className="model-notice-icon">
                    <AlertTriangle size={15} strokeWidth={2} />
                </span>
                <div>
                    <div className="model-notice-title">Running on generic model</div>
                    <div className="model-notice-desc">
                        No trained PCB model found. Detections may not match real PCB defects.
                        Place <code>pcb_model.pt</code> inside <code>models/</code> and restart
                        the backend to enable PCB-specific inference.
                    </div>
                </div>
            </div>
            <button
                className="model-notice-close"
                onClick={() => setDismissed(true)}
                aria-label="Dismiss"
            >
                <X size={14} />
            </button>
        </div>
    )
}
