import { pcbPaths } from './pcbPaths'

export default function PCBBackground() {
    return (
        <div
            style={{
                position: 'fixed',
                inset: 0,
                width: '100%',
                height: '100%',
                pointerEvents: 'none',
                zIndex: 0,
            }}
        >
            {/* The static inverted base image for the grey circuit traces */}
            <div
                style={{
                    position: 'absolute',
                    inset: 0,
                    width: '100%',
                    height: '100%',
                    backgroundImage: 'url("/pcb-circuit.svg")',
                    backgroundSize: 'cover',
                    backgroundPosition: 'center',
                    backgroundRepeat: 'no-repeat',
                    filter: 'invert(1)',
                    opacity: 0.06,
                }}
            />

            {/* The dynamic SVG overlay for the animated glowing green dots/signals */}
            <svg
                viewBox="0 0 2100 2100"
                preserveAspectRatio="xMidYMid slice"
                style={{
                    position: 'absolute',
                    inset: 0,
                    width: '100%',
                    height: '100%',
                    opacity: 0.8,
                }}
            >
                <defs>
                    {/* Glowing green line gradient for the signal */}
                    <linearGradient id="signalGlow" x1="0%" y1="0%" x2="100%" y2="0%">
                        <stop offset="0%" stopColor="#22c55e" stopOpacity="0" />
                        <stop offset="50%" stopColor="#4ade80" stopOpacity="0.8" />
                        <stop offset="100%" stopColor="#22c55e" stopOpacity="0" />
                    </linearGradient>

                    {/* Filter for actual neon glow effect */}
                    <filter id="neon">
                        <feGaussianBlur stdDeviation="15" result="coloredBlur" />
                        <feMerge>
                            <feMergeNode in="coloredBlur" />
                            <feMergeNode in="SourceGraphic" />
                        </feMerge>
                    </filter>
                </defs>

                {/* Render a subset of traces and animate a "dash" along them */}
                {pcbPaths.map((d, i) => {
                    // Only animate about half of the paths so it's not overly chaotic
                    if (i % 2 !== 0) return null

                    // Randomize animation duration and delay per path
                    const duration = 4 + Math.random() * 6 // 4s to 10s
                    const delay = Math.random() * 5        // up to 5s offset

                    return (
                        <g key={i}>
                            {/* The invisible track, just provides the animated stroke */}
                            <path
                                d={d}
                                fill="none"
                                stroke="url(#signalGlow)"
                                strokeWidth="60"
                                strokeLinecap="round"
                                filter="url(#neon)"
                                transform="matrix(0.1, 0, 0, -0.1, 0, 2100)"
                                style={{
                                    strokeDasharray: '400 6000', // A short 400px dash, followed by 6000px gap
                                    animation: `dashAnim ${duration}s linear infinite`,
                                    animationDelay: `-${delay}s`
                                }}
                            />
                        </g>
                    )
                })}
            </svg>

            <style>{`
                @keyframes dashAnim {
                    0% {
                        stroke-dashoffset: 6400;
                    }
                    100% {
                        stroke-dashoffset: 0;
                    }
                }
            `}</style>
        </div>
    )
}
