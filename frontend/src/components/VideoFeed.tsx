import { useEffect, useState } from 'react'
import { useVideoStream } from '../hooks/useVideoStream'
import { useStore } from '../store'
import { camerasApi } from '../api/client'
import { Play, Square } from 'lucide-react'

interface Props {
  cameraId: number
  cameraName?: string
  compact?: boolean
}

export default function VideoFeed({ cameraId, cameraName, compact }: Props) {
  const { frame, connected, clearFrame } = useVideoStream(cameraId)
  const metrics = useStore(s => s.metrics[cameraId])
  const clearMetrics = useStore(s => s.clearMetrics)

  const [running, setRunning] = useState(false)
  const [toggling, setToggling] = useState(false)

  useEffect(() => {
    camerasApi.status(cameraId).then(s => setRunning(s.running)).catch(console.error)
  }, [cameraId])

  const toggle = async () => {
    setToggling(true)
    try {
      if (running) {
        await camerasApi.stop(cameraId)
        setRunning(false)
        clearFrame()
        clearMetrics(cameraId)
      } else {
        await camerasApi.start(cameraId)
        setRunning(true)
      }
    } finally {
      setToggling(false)
    }
  }

  return (
    <div className="relative bg-cyber-surface rounded border border-cyber-border overflow-hidden">
      {/* Status badge */}
      <div className="absolute top-2 left-2 z-10 flex items-center gap-2">
        <span className={`font-mono text-xs px-2 py-0.5 rounded border ${
          connected
            ? 'bg-cyber-cyan/10 border-cyber-cyan/40 text-cyber-cyan'
            : 'bg-white/5 border-cyber-border text-cyber-muted'
        }`}>
          {connected ? '● LIVE' : '◌ OFFLINE'}
        </span>
        {cameraName && (
          <span className="font-mono text-xs text-cyber-muted bg-cyber-bg/60 px-2 py-0.5 rounded">
            {cameraName}
          </span>
        )}
      </div>

      {/* On/off toggle */}
      <button
        onClick={toggle}
        disabled={toggling}
        title={running ? 'Stop' : 'Start'}
        className={`absolute top-2 right-2 z-10 flex items-center gap-1 font-mono text-xs px-2 py-0.5 rounded border transition-colors disabled:opacity-50 ${
          running
            ? 'border-cyber-red/40 text-cyber-red hover:bg-cyber-red/10'
            : 'border-cyber-green/40 text-cyber-green hover:bg-cyber-green/10'
        }`}
      >
        {running ? <Square size={11} /> : <Play size={11} />}
        {toggling ? '…' : running ? 'STOP' : 'START'}
      </button>

      {/* Video */}
      {frame ? (
        <img src={frame} alt="live feed" className="w-full max-h-[70vh] h-auto object-contain mx-auto block" />
      ) : (
        <div className={`flex items-center justify-center text-cyber-muted font-mono text-sm ${compact ? 'h-48' : 'h-96'}`}>
          {connected ? 'Waiting for frames…' : 'Camera not streaming'}
        </div>
      )}

      {/* Metrics overlay (compact mode shows fewer) */}
      {metrics && !compact && (
        <div className="absolute bottom-0 left-0 right-0 bg-cyber-bg/70 backdrop-blur px-3 py-2 flex gap-4 text-xs font-mono">
          <span className="text-cyber-cyan">P: {metrics.people}</span>
          <span className={metrics.gun > 0 ? 'text-cyber-red' : 'text-cyber-muted'}>GUN: {metrics.gun}</span>
          <span className={metrics.knife > 0 ? 'text-cyber-red' : 'text-cyber-muted'}>KNIFE: {metrics.knife}</span>
          <span className={metrics.running > 0 ? 'text-cyber-orange' : 'text-cyber-muted'}>RUN: {metrics.running}</span>
          <span className={metrics.loiter > 0 ? 'text-cyber-orange' : 'text-cyber-muted'}>LOIT: {metrics.loiter}</span>
          <span className={metrics.fall > 0 ? 'text-cyber-red' : 'text-cyber-muted'}>FALL: {metrics.fall}</span>
          <span className="ml-auto text-cyber-muted">{metrics.fps} FPS</span>
        </div>
      )}
    </div>
  )
}
